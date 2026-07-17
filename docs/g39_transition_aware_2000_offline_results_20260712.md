# [Execution target G39/H1: Transition-aware ACT 2000-epoch 完整离线结果]

日期：2026-07-12。目标 worktree 为
`/home/pingfan/Excavator_real_stack_e52_deadlock_eval`，分支
`codex/e52-offline-deadlock-gate`，基线 HEAD 为
`0fab67eda7e449b70622b65afc7ada01142f56e5e5`。主 checkout 没有修改。

本轮严格沿用 episode-77 扩展 train split、冻结 validation `[94, 91, 84, 74, 92]`，禁止
105..109，policy action scale 为 identity `[1, 1, 1, 1]`。原始 HDF5 未修改，也没有连接
Jetson、现场 TCP 或执行部署。

## [Execution target G39/H1: 训练与 provenance]

split artifact：
`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g37_transition_aware_formal/train_val_split.yaml`

- SHA256：`14ab7e9e67382646bc5b922ac48ab31eecd41217caa9366f0259f85a0c2844f6`
- train：20 episodes（formal 19 + episode 77）
- validation：5 episodes，未变化
- held-out：105..109，未使用、未校准、未评估

| 候选 | 输入/辅助监督 | best epoch | best val loss | checkpoint SHA256 |
|---|---|---:|---:|---|
| A / transition boundary | qpos；transition-window deadzone loss | 1550 | 0.1032421142 | `5eb39bbaa8f296de8c2e15191d73cedee703ccc4d082a06dc3e7b12b42965520` |
| B / transition phase qvel | qpos+qvel；当前 query intent auxiliary loss | 150 | 0.0939745549 | `672939a1f7409ee0e9ec3a7c9026ddd684e64f8f676f7033c28669a9dc254c58` |

配置分别为：

- [A config](/home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed/testbed/configs/act_real_gmsl_eye2_g38_transition_boundary_2000.yaml)，SHA256 `21a79a1abce861a88f270358ff09286a7648b03e4ff18e14b6a2f9e8be1742ea`
- [B config](/home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed/testbed/configs/act_real_gmsl_eye2_g38_transition_phase_qvel_2000.yaml)，SHA256 `a606802db954484ee55fad47975cad1543cb6485e631ade4e53733856ff8f10b`

两者都真正跑满 2000 epoch，并保留了 epoch 0、499、999、1499、1999 及 best checkpoint。

## [Execution target G39/H2: 全量 open-loop 与轨迹 liveness]

评估覆盖固定 formal 24 episodes、16529 steps；没有把 episode 77 作为最终比较集，也没有
触碰 held-out。完整 open-loop artifact：

- [A collection summary](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_boundary_2000_formal/open_loop_all24/collection_summary.json)
- [B collection summary](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_phase_qvel_2000_formal/open_loop_all24/collection_summary.json)
- [trajectory liveness report](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_2000_formal/trajectory_deadzone_liveness/trajectory_deadzone_liveness_report.json)，SHA256 `75127770a733fa508cdf62cbcd7ea938eb47e2ae4b7477a5a45a038a2ec1e691`

| pipeline | open-loop MAE | raw hit-20 | assist hit-20 | assist persistent-20 | assist underconfidence | assist extra/wrong |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.05158 | 96.929% | **99.272%** | **99.158%** | 1.072% | 12.784% |
| B | 0.06276 | 96.216% | 98.131% | 98.032% | 1.853% | 17.128% |
| raw ACT + assist reference | — | — | 98.943% | 98.836% | 2.121% | 11.478% |

A 的轨迹 liveness 略高于 raw ACT + assist，但 extra/wrong 已高于 reference；B 在 liveness 和
extra/wrong 两项都更差。2000 epoch 没有把连续动作变成更稳定的“坚定启动”。

## [Execution target G39/H3: recursive state-hold deadlock/hidden]

validation 共 48 个 transition anchors，hold horizon 20；每个 anchor 都从 reset 重放，冻结
qpos/image 并将 qvel 置零。startup anchor 不是手工指定，而是每个 episode 的第一个
`inactive -> expert-effective` 转换。五个 validation episode 的 startup 主轴均为 bucket+，
expert command 约 `0.410..0.427`，A/B 在 raw 和 assist 下均为 5/5 立即恢复。

| 候选 | assist | recovered | deadlocked | hidden | unexpected anchors |
|---|---|---:|---:|---:|---:|
| A | disabled | 40/48 | 8 | 1 | 3 |
| A | enabled | **46/48** | 2 | **1** | 8 |
| B | disabled | 42/48 | 6 | 0 | 4 |
| B | enabled | 44/48 | 4 | **1** | 10 |

官方 complete gate 要求 assist `recovered >= 46` 且 `hidden == 0`。A 只满足 recovered，仍有
1 个 hidden 和 2 个 deadlock；B 两项都没有满足。因此两者都不能替换 raw ACT + assist。

artifact：

- [A state-hold run](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_boundary_2000_formal/state_hold_val5_h20/run_summary.json)，SHA256 `1fd9679f4c578a2f3f282ae95e9ebd597635d3169d6d77e6696da7a7982c6d6f`
- [B state-hold run](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_phase_qvel_2000_formal/state_hold_val5_h20/run_summary.json)，SHA256 `db8ab476998d9a459f307fd6c7d569068345d2bad86f1c340547553e98194980`

## [Execution target G39/H4: tail、release、gohome 与 execution monitor]

| 候选 | tail extra frames | release extra frames | clip/nonfinite | max active effective axes |
|---|---:|---:|---|---:|
| A | 34 | 401 | 0 / 0 | 2 |
| B | 133 | 568 | 0 / 0 | 2 |

正式 HDF5 没有完整 `gohome_eligible_label` / `tail_idle_mask`，所以 gohome false-positive
不能从现有数据估计；报告明确标记 `estimable=false`，没有假装通过。

现有 execution-response sidecar 只用于独立 monitor replay，不用于训练 retry 策略：validation
48/48 有响应，formal train+validation 236/240 有响应、4 个 stalled candidates；但 sidecar
没有 policy intent、operator correction 或已确认液压故障标签，retry precision 不可估计。

- [execution monitor report](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_2000_formal/execution_monitor/execution_monitor_eval.json)，SHA256 `501218c7fd878d90364e7299c5a5f399fd7e98459bafa369970b39efd82752b5`

## [Execution target G39/H5: fixed-qpos / multi-FPV gate]

对 A/B 各做了 `episode_74 <- image_91` 与 `episode_92 <- image_74` 的 `nearest_qpos` replay。
两组的 qpos matching p95 normalized distance 分别为 `2.61` 和 `3.88`，匹配质量过差，按协议
只能标记为 `inconclusive`，不能拿它们证明模型通过或失败。该结果也不会用于调参。

## [Execution target G39/H6: complete offline gate 与最终选择]

两份 complete report 均包含 open-loop window rows、state-hold summary、tail/release、gohome
estimability、execution-monitor replay、split/hash 和 fixed-qpos artifacts：

- [A complete report](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_boundary_2000_formal/complete_offline_report/complete_offline_report.json)，SHA256 `f881a549b8b2039e55adb8cfba208e283d4b18dbb05b8543f3143bc29eb22a47`
- [B complete report](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g38_transition_phase_qvel_2000_formal/complete_offline_report/complete_offline_report.json)，SHA256 `02d4403a465f3fef2aabc8e26c75da721017e5268f0e912cde1c9dbe4738e3fe`

最终选择：**A 是本轮较好的候选，但 A/B 都拒绝进入 control；继续使用 raw ACT + mechanical assist 作为回退/reference。**

## [Execution target G39/H7: callback 与反思]

1. transition-window deadzone loss 确实减少了 underconfidence，A 的 assist hit-20 超过 raw
   ACT + assist；但它同时引入 hidden anchor 和更多 unexpected effective motion，说明“越过
   死区”不能只靠幅值 promote loss 保证闭环安全。
2. B 的 qvel + intent auxiliary 使 validation loss 更低，却让 open-loop MAE、extra/wrong、
   state-hold deadlock 都变差；这证明不能用 val loss 或辅助分类准确度替代执行闭环指标。
3. 两个候选的 startup transition 都通过，但 mid-cycle 仍有 2/4 个 deadlock；当前真正未解决
   的不是“模型永远不会启动”，而是状态推进后的局部动作保持/阶段切换仍可能落入吸引态。
4. 因此本轮没有部署、没有使用 held-out 选择阈值，也没有继续扫旧 gate 阈值。下一轮若继续，
   应直接针对 mid-cycle 的 causal progress/transition state 设计训练或评估，而不是继续扩大
   2000-epoch 连续回归或 qvel auxiliary 的权重搜索。

## [Execution target G39/H8: verification]

- `PYTHONPATH=.:testbed pytest -q`：**438 passed，4 subtests passed，16 warnings**。
- `git diff --check`：通过。
- 仓库级 `ruff check testbed/testbed testbed/tests` 仍报告 172 个既有全树 lint 问题；本轮没有
  用自动 formatter 改写无关文件，也没有把该全树基线问题误报成模型实验失败。
