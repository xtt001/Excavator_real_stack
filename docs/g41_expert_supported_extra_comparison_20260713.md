# [Execution target G41/H1: Expert-supported extra axis 对比]

日期：2026-07-13。目标是用专家数据重新解释 `extra/wrong`，并在同一 eye2 输入
契约下比较 G38 A 与 raw ACT baseline。未修改现场 runtime、未连接 Jetson、未读取
held-out 105..109，原始 HDF5 未修改。

## [Execution target G41/H1: Fair baseline 与 provenance]

本轮使用的 baseline 是与 A 同为 `video4/video5` 的
`baseline_qpos_no_transform_eye2`，而不是四相机的
`baseline_qpos_no_transform`。四相机 baseline 与 A 的直接比较不作为结论。

- baseline bundle：`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_eye2`
- baseline collection summary SHA256：`d2a070d20718290c57ce2ed04ae3d7c3eeaa75926c457b1349195355fb7320d0`
- split SHA256：`14ab7e9e67382646bc5b922ac48ab31eecd41217caa9366f0259f85a0c2844f6`
- train-only joint support 包含 episode 77；validation `[94,91,84,74,92]` 冻结。

## [Execution target G41/H2: Expert joint-action support]

当前 train split 中观察到的有效联合方向次数：

| joint direction | train count |
|---|---:|
| boom+ + bucket- | 945 |
| boom+ + bucket+ | 59 |
| boom- + bucket- | 30 |
| boom- + bucket+ | 22 |
| boom+ + swing- | 19 |
| boom- + swing- | 4 |

因此 boom/bucket 同时动作不是非法模式。严格指标把“当前专家这一帧尚未越过死区、但
同方向动作已经接近死区”的 policy 输出也标成 extra。

## [Execution target G41/H3: Refined extra 定义]

对每个专家有效帧中的额外 policy 方向，分别统计：

- `strict extra`：policy 有效方向不在专家有效方向集合中；
- `same-sign near`：专家原始 action 与额外方向同号，且幅值至少达到该方向死区的 0.5；
- `joint-supported`：当前专家有效方向与额外方向的组合在 train split 中出现过；
- `temporal-supported`：同方向专家有效动作在当前帧 ±8 tick 内出现；
- `unbacked`：以上三类都不满足。这个指标只用于离线解释，不改变 runtime command。

## [Execution target G41/H4: 全 formal 24 episodes 对比]

| model / variant | open-loop MAE | hit-20 | persistent-20 | strict extra | same-sign near | train-joint-supported | unbacked |
|---|---:|---:|---:|---:|---:|---:|---:|
| A raw | 0.05158 | 96.929% | 96.401% | 799 / 11788 = 6.78% | 83.23% | 99.87% | 0 |
| eye2 baseline raw | 0.04728 | 96.791% | 96.286% | 570 / 11788 = 4.84% | 95.96% | 100.00% | 0 |
| A + assist | — | **99.272%** | **99.158%** | 1507 / 11788 = 12.78% | 68.28% | 96.75% | 37 = 2.46% |
| eye2 baseline + assist | — | 98.690% | 98.637% | 1328 / 11788 = 11.27% | 78.39% | 96.84% | 23 = 1.73% |

A 的 assist liveness 比 eye2 baseline 高约 `0.58` 个百分点，但 strict extra 高约
`1.52` 个百分点。重要的是，A 的额外方向中 `96.75%` 属于 train 中已观察过的联合
动作组合；所以这 12.78% 不能整体解释成错误动作。

## [Execution target G41/H5: 冻结 validation 对比]

在 5 个 validation episodes 上：

| model + assist | hit-20 | persistent-20 | strict extra | unbacked extra |
|---|---:|---:|---:|---:|
| A | **97.969%** | **97.747%** | 279 / 2365 = 11.80% | 26 |
| eye2 baseline | 97.710% | 97.563% | 247 / 2365 = 10.44% | 23 |

A 的 validation liveness 有小幅提升，但 strict extra 和 unbacked extra 也略高；这不是
“所有指标都变好”，而是动作启动/持续性与额外轴之间的明确 trade-off。

## [Execution target G41/H6: 同一 recursive state-hold 对比]

两者都用同一个 48-anchor、hold horizon 20、identity action scale contract：

| model | assist | recovered | deadlocked | hidden | startup |
|---|---|---:|---:|---:|---:|
| A | disabled | 40/48 | 8 | 1 | 5/5 |
| A | enabled | **46/48** | **2** | 1 | 5/5 |
| eye2 baseline | disabled | 35/48 | 13 | 5 | 3/5 |
| eye2 baseline | enabled | 43/48 | 5 | 3 | 5/5 |

artifact：

- [trajectory comparison](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260713/g41_refined_extra_comparison/trajectory_deadzone_liveness_eye2/trajectory_deadzone_liveness_report.json)，SHA256 `0a4f53325a49d25c1e80b4ffbd0aa130cc4d224eab3d2a02b0436b4e6de10865`
- [baseline eye2 state-hold](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260713/g41_refined_extra_comparison/baseline_eye2_state_hold_val5_h20/run_summary.json)，SHA256 `b2fb4b7b1606fa1eab0f9a5e66086d8f7c12d9227298e6cc6f5e6699cb762e70`

## [Execution target G41/H7: 判断]

结论分两层：

1. **如果使用原始 strict extra 指标，A 没有提升**：A 多出了约 1.5 个百分点的有效额外帧。
2. **如果按照专家支持度解释，A 的主要收益是真实的**：A 的 assist liveness 更高，
   state-hold recovered 从 `43/48` 提升到 `46/48`，deadlock 从 `5` 降到 `2`，hidden 从
   `3` 降到 `1`；而大多数 boom/bucket extra 都属于专家已经出现过的联合动作模式。

因此，A 不应再因为 `12.78% strict extra` 被整体判定为严重错误；更准确的描述是：

> A 用少量额外轴动作换取了更坚定、更少 deadlock 的执行。这个 trade-off 在当前专家
> 数据下有支持，但 A 仍有 1 个 hidden anchor 和 2 个 deadlock，尚不能进入 control。

下一步 gate 应把 `unbacked extra`、同方向近死区意图和已观察 joint mode 分开统计，不能再
把所有额外 boom/bucket 统一判错。
