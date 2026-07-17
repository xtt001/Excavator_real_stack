# [Execution target G32/H1: 全轨迹死区可执行性评估]

## [Execution target G32/H1: 测试定义]

本轮不选择单一 startup anchor，而是扫描每个 expert-effective 帧中的每个轴/方向。
对当前帧以及未来 4、8、20 个 20 Hz tick 分别检查：

- 是否越过该方向机械死区（`hit_hN`）；
- 越过后是否连续保持 2 tick（`persistent_hN`）；
- 当前 signed margin 是否为负；
- 当前帧是否出现 expert 未要求的有效轴/方向。

多轴动作按轴/方向独立计分，不把 `boom+ + bucket+` 当成非法模式。机械 assist
只作为顺序一致的 counterfactual command variant，不改变 checkpoint，也不改变
policy action domain。policy action scale 固定为 `[1, 1, 1, 1]`。

数据范围是正式 train-ready manifest 的 24 个 episode、16,529 steps、11,788 个
expert-effective frames、13,058 个轴/方向 opportunities；episode 105..109 未使用。
死区表来自 direct mechanical output artifact，SHA-256 为
`780fdc1c24c17b4bf4d3c67f7b07e7237e68cb1e0c7264be311a1056346a54c1`。

## [Execution target G32/H1: 全局结果]

| variant | 当前同向有效 | 4 tick 内命中 | 20 tick 内命中 | 当前 underconfidence | 20 tick 零命中 segment |
|---|---:|---:|---:|---:|---:|
| raw ACT | 87.49% | 89.57% | 95.02% | 12.34% | 18 / 228 |
| raw ACT + mechanical assist* | 97.71% | 98.09% | 98.94% | 2.12% | 4 / 228 |
| G28 raw | 95.24% | 95.80% | 96.68% | 4.61% | 14 / 228 |
| G28 + mechanical assist* | **99.03%** | **99.17%** | **99.40%** | **0.83%** | **2 / 228** |

`*` assist 结果是离线顺序 counterfactual，不能替代 recursive state-hold 结果。

G28 相比 raw ACT 明显降低了“方向正确但幅度不够”的比例，说明辅助监督确实增强了
动作信心；但它没有消除长时间的零动作吸引态。加入 assist 后平均命中率接近 99%，
但仍有两个完整有效动作段在 20 tick 内完全没有 boom+ 命令。

代价也被保留下来：按“expert 已经有效的帧”为分母，G28 + assist 的额外/错误有效
轴帧率为 `14.82%`，高于 raw ACT + assist 的 `11.48%`。因此不能只把 assist 或全局
增幅当成模型修复；训练必须同时提高目标轴 margin 和抑制非目标轴。

## [Execution target G32/H1: 分轴诊断]

全数据按 opportunity 加权的关键结果：

| variant | 方向 | 当前有效 | 当前 underconfidence | 4 tick 命中 |
|---|---|---:|---:|---:|
| raw ACT | bucket+ | 90.2% | 9.8% | 92.2% |
| G28 raw | bucket+ | 94.7% | 5.3% | 95.4% |
| G28 + assist | bucket+ | **99.7%** | **0.3%** | **99.7%** |
| raw ACT | boom+ | 82.5% | 17.5% | 84.7% |
| G28 raw | boom+ | 93.2% | 6.8% | 93.9% |
| G28 + assist | boom+ | **95.9%** | **4.1%** | **96.4%** |
| raw ACT | swing- | 83.5% | 16.5% | 85.4% |
| G28 raw | swing- | 97.6% | 2.4% | 97.6% |

这说明问题不是 bucket 这个主轴本身没有学到。bucket 在 G28 中已经明显改善；剩余
风险是某些视觉/姿态状态下连续动作幅度突然塌到死区以下，尤其是 boom+，其它轴也
存在同类现象。

## [Execution target G32/H1: 仍然失败的完整动作段]

G28 + assist 的两个 20 tick 零命中段是：

| episode | 轴方向 | expert-effective 区间 | G28 raw boom 输出 |
|---|---|---:|---:|
| episode_94 | boom+ | 474..502 | 最大约 `+0.048` |
| episode_98 | boom+ | 542..560 | 约 `-0.016`，一直为反向/近零 |

boom+ 的机械死区是 `+0.259`。这两段不是“晚了几帧”，而是整段没有可执行的主轴
命令，因此与真机的“某一时刻动不了，后面也动不了”机制一致。episode_74 的部分
boom+ 段可以在更长窗口后逐渐变大，但仍属于低余量、容易受 state-hold 影响的风险段。

已有 recursive state-hold 完整结果仍是最终安全约束：G28 + assist 为 `45/48`，
还有 3 个 deadlock，未达到替换 raw ACT + assist 的 promotion gate。全轨迹测试解释了
为什么平均命中率很高仍不能放行：平均值掩盖了少数完整零命中段。

## [Execution target G32/H1: 对模型的调整建议]

1. **把训练监督从 transition/anchor 扩展为密集有效区间监督。** 对每个 train-fold
   expert-effective 帧和每个目标轴/方向加入 signed-margin hinge：

   `relu(deadzone + reviewed_margin - sign(expert) * policy_action)`。

   初始 `reviewed_margin` 可沿用当前机械 assist 的 `0.02`，但只在 train fold 选择，
   不能用 validation 或 episode 105..109 调参。

2. **加入整段 coverage loss，而不是只要求某个起点最终恢复。** 对连续有效段统计
   4 tick 内命中率和最长零命中 run；当一个有效段全部低于死区时，提高该段 loss。
   这样才能直接压制 episode_94/98 这类吸引态。

3. **保留连续动作作为唯一 command source。** intent/effect head 只做辅助监督和
   loss eligibility，不在运行时 hard argmax 覆盖 action；不恢复 E52 gate，不使用
   global action scale，不把动作压成单轴。

4. **做按轴/方向的 train-only reweighting。** 当前最需要增加的不是新的联合模式，
   而是训练中低 margin、长零命中段的权重，优先覆盖 boom+、boom-、bucket+ 等真实
   失败方向；同时保留 inactive/extra-axis loss，避免 assist 后额外有效动作上升。

5. **监督必须作用在 temporal aggregation 之后的 direct policy output。** 评测看的是
   最终发送域动作；只在 chunk 内某个 query 输出很大、但聚合后低于死区，仍然会在真机
   上失败。训练 loss 和离线 metric 要使用同一个 direct-output domain。

6. **新模型的放行标准改为“平均指标 + 最坏动作段 + state-hold”。** 至少要求：
   4 tick target hit、当前 margin、零命中 segment 数、extra/wrong 和 recursive
   deadlock 同时不劣于 raw ACT + assist；不能只看 MAE 或全局平均命中率。

结论：G28 的方向判断和平均动作信心已经比 raw ACT 好，但还没有解决根本的“长段
主轴命令塌陷”。下一轮应训练 dense per-axis margin/coverage，而不是继续增加模式
分类器或 gate。当前现场基线仍应保持 raw ACT + mechanical assist。

## [Execution target G32/H1: 可复查产物]

- 报告：[trajectory_deadzone_liveness_report.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/g32_trajectory_deadzone_liveness_20260712/trajectory_deadzone_liveness_report.json)
- 逐 opportunity：[opportunity_rows.csv](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/g32_trajectory_deadzone_liveness_20260712/opportunity_rows.csv)
- 分段：[segment_summary.csv](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/g32_trajectory_deadzone_liveness_20260712/segment_summary.csv)
- 报告 SHA-256：`0f386f62e2f47654ab9de6fc382fcab2697797c67a6a12bfeeefadbe332f51df`

Boundary decision: new focused owner (`trajectory_deadzone_liveness.py` plus CLI).
Reason: dense trajectory scoring is a reusable evaluation capability, not state-hold
or policy-action runtime ownership.
Verification: focused unit tests, Ruff, compileall, and the full formal 24-episode run.
