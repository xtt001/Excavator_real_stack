# [Execution target G36/H1: “坚定启动主轴”数据审计契约]

本轮不训练新 checkpoint，也不修改原始 HDF5；目标是先回答“专家在什么
条件下从 idle 进入有效动作”，再决定如何训练。审计使用正式 24 episode：
19 个 train、5 个 validation，共 16,529 steps。held-out 105..109 未读取、
未校准、未评估。

使用的死区仍是现场确认值：swing +0.661/-0.721、boom +0.259/-0.357、
stick +0.500/-0.500（task-structural zero）、bucket +0.408/-0.508。

审计 owner 是
[action_start_distribution.py](/home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed/testbed/policies/action_start_distribution.py)，
CLI 是
[audit_action_start_distribution.py](/home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed/testbed/cli/audit_action_start_distribution.py)。

# [Execution target G36/H2: 第一条动作到底是什么]

结果非常明确：在这 24 条正式 episode 中，**第一条越过死区的动作 24/24
都是 bucket+**。

| split | episode 数 | 第一有效动作 | step 中位数（P10–P90） | 越过死区时的动作比值中位数（P10） |
| --- | ---: | --- | ---: | ---: |
| train | 19 | bucket+ 19/19 | 53（32.2–70.2） | 1.0346（1.0112） |
| validation | 5 | bucket+ 5/5 | 38（29.2–62.6） | 1.0203（1.0081） |
| all | 24 | bucket+ 24/24 | 51.5（29.6–67.1） | 1.0311（1.0077） |

这说明当前任务的“启动主轴”不是抽象的任意一根轴，而是可以从真实数据
定义为**第一段 bucket 正向启动**。但这不意味着运行时应硬编码 bucket：
它是本操作风格的统计先验；若操作流程改变，模型仍应从视觉和时序上下文
预测启动轴。

# [Execution target G36/H3: 启动不是稀有动作，稀有的是死区边界]

专家一旦越过死区，通常会持续很久；困难集中在第一次越过的那一两个 tick。

| axis/direction | transitions | 持续 ≥4 tick | 持续 ≥10 tick | onset 比值中位数 | 前 5 tick 峰值比值中位数 | 该轴 idle 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| swing+ | 24 | 100.0% | 100.0% | 1.021 | 0.984 | 347 |
| swing- | 25 | 100.0% | 100.0% | 1.016 | 0.986 | 115 |
| boom+ | 44 | 97.7% | 93.2% | 1.045 | 0.949 | 47 |
| boom- | 31 | 100.0% | 96.8% | 1.050 | 0.927 | 101 |
| bucket+ | 54 | 98.1% | 98.1% | 1.035 | 0.961 | 66.5 |
| bucket- | 50 | 98.0% | 94.0% | 1.028 | 0.964 | 43.5 |

onset_action_ratio 是“实际动作 / 该方向死区”。因此启动的典型形状是：
前几 tick 已经接近死区但仍不动（例如 bucket+ 的中位数峰值为 0.961），
下一 tick 只刚刚超过死区（中位数 1.035，P10 约 1.008），然后保持有效。
这正好解释了真机上的“动作意图有了，但启动不坚定”：连续回归只要把
1.008 略微平均到 0.99，就会从“能启动”变成“完全不动”。

死区附近的全量计数也支持这个判断。以 [0.9,1.0) 为“接近但未越过”，
以 [1.0,1.1) 为“刚越过”：

| direction | 有效 steps | 接近但未越过 | 刚越过 |
| --- | ---: | ---: | ---: |
| boom+ | 2,035 | 118 | 127 |
| boom- | 1,109 | 64 | 79 |
| bucket+ | 2,589 | 151 | 199 |
| bucket- | 2,846 | 259 | 267 |
| swing+ | 2,216 | 169 | 210 |
| swing- | 2,263 | 203 | 804 |

因此问题不是“数据里没有主轴动作”，而是**启动样本只有 228 次 transition，
且每次 transition 的判别间隔非常窄**。随机按 frame 采样时，transition 起点
只占约 228 / 16,529 = 1.38%；每条 episode 的第一启动更只有 24 / 16,529。

# [Execution target G36/H4: 多轴支持缺口与 qpos/qvel 可辨识性]

## 正式 train/validation 的 joint-mode 分布

正式 train 中 boom+|bucket+ 仍为 **0**，validation 中有 29 steps：

| joint mode | train | validation |
| --- | ---: | ---: |
| boom+ \| bucket+ | **0** | **29** |
| boom+ \| bucket- | 867 | 314 |
| boom+ only | 637 | 169 |
| bucket+ only | 2,094 | 459 |

这不是“多轴动作不允许”，而是当前冻结 train split 对该正向联合模式没有
support。G30 已从现有非 held-out episode 77 观察到 59 个
boom+|bucket+ steps，因此不需要重新采集；后续可重划 train-only split
纳入 episode 77，同时保留 episode 94 作 validation，不改变当前 baseline
的冻结评测契约。

## qpos/qvel 不能单独决定启动

按 train quantile bins 统计相同 qpos/qvel bin 内的 idle/pos/neg 分布：

- swing 加权 entropy 0.222 bits；
- boom 0.281 bits；
- bucket 0.373 bits，其中 5 个 bin 的 entropy ≥1 bit；
- 例如 bucket 的一个 bin 有 idle=6,pos=36,neg=20，另一个有
  idle=233,pos=383,neg=52。

所以 qpos/qvel 是有用条件，但不是“到某个 qpos 就必然启动”的充分条件。
第一条 bucket+ 的 onset qpos 在 train 中约为 0.2355（P10–P90：
0.2295–0.2514），onset qvel 中位数约 -0.0018；相同附近仍有大量
idle frame。真正可辨识的信号是**视觉状态 + 时序阶段 + 逐步 ramp**，不是
单个瞬时 qpos。

本 artifact 没有把图像 hash 或视觉 embedding 当作标签，因此
image_ambiguity_measured=false；这里不会把未测量的视觉分布猜成 OOD。

# [Execution target G36/H5: 根本训练调整]

结论不是再加一个 E52 gate，而是把训练目标从“每帧回归动作”改成
“连续动作 + 启动阶段建模”：

1. **从现有 action 序列派生 phase 标签**：idle → ramp-near-boundary →
   start-pos/start-neg → hold → release。只在真实 transition 周围取
   causal window（前 5 tick、启动后 4–10 tick），按 episode 平衡采样；不要
   让 1.38% 的 transition 被普通 idle frame 淹没。
2. **连续 ACT 仍是 command source of truth**。增加 axis/direction 的
   start/hold/release 辅助头，用 soft probability 提供 eligibility/表征
   学习；不使用 per-frame hard argmax、projection、gate，也不把
   action_scale 作用到模型动作。
3. **对启动做不对称的 boundary loss**：启动窗口内重点惩罚
   signed_action < threshold，对“略高于死区”不作同等惩罚；保持 ramp
   和 operator style。不要把全 chunk 的未来标签复制给当前 query（G34
   的问题），也不要把 current-step-only 监督做成全局保守（G35 的问题）。
   可用排序/softplus 约束“onset 高于 threshold、前一窗口仍低于 threshold”，
   而不是固定给每个动作加一个未经数据支持的 +0.02。
4. **输入加入因果时序上下文**：短时视觉 token、qpos/qvel delta 和 ramp
   斜率；不把 previous command 作为 policy core 的 phase cue。若要使用
   execution history，应放在独立 monitor。
5. **补齐联合模式 support**：用现有 episode 77 重划 train-only split，
   并单独报告 boom+|bucket+ 的 precision/recall；不能以 MAE 代替
   joint-mode liveness。

# [Execution target G36/H6: 离线验收与本轮 callback]

下一轮候选模型必须在固定 validation 上同时通过：

- 第一启动：5/5 个 validation episode 在 transition window 内使专家主轴
  越过死区，并保持 ≥4 tick；
- 全部 228 个 transition 的 target-axis underconfidence 降低；
- wrong/extra、tail、release、gohome 不回退；
- 48-anchor recursive state-hold 中 deadlock/hidden=0，并超过现有
  raw ACT + mechanical assist reference 后才允许触碰 held-out。

本轮仅完成分布审计和训练设计，**没有声称已经提升模型成功率**。现有
G35 complete offline 结果仍是参考：raw ACT + assist 在 validation
state-hold 为 43/48 recovered、5 deadlock；因此 G36 的训练目标必须
直接减少“主轴未越过死区”，不能只改善 MAE 或分类准确率。

Artifacts：

- [action_start_distribution_report.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g36_action_start_distribution/action_start_distribution_report.json)，SHA256 f2490a3b8bf85d6522ee125fa45df75c7c76293cc0dbac0a835919cdeac84807；
- [transition_rows.csv](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g36_action_start_distribution/transition_rows.csv)，SHA256 45c9a7a98780655bdb1f8dbdc3bd8b4212e77bc8b1f74689eaa4df825efa3a37；
- [first_transition_rows.csv](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g36_action_start_distribution/first_transition_rows.csv)，SHA256 9a2dc94472c664d37d658501054bf9b73b68588f946ec490202b02f0f4900ed2；
- [transition_summary.csv](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g36_action_start_distribution/transition_summary.csv)，SHA256 7568f364f0b08f495b2a60f9356aa4cec49c657d94b2aa4229dfeac4057c6944。

Source HDF5 未写入；本轮验证：436 passed, 16 warnings, 4 subtests
passed；changed-owner Ruff、compileall、git diff --check 均通过。
