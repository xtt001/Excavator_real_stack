# [Execution target G43/H1: 可回退短时序 ACT 输入实验]

本轮验证了一个可回退的改动：让 ACT 在离线训练和推理时看到最近 4 帧图像的因果窗口，
而不是只看当前一帧。改动只在独立的 G43 配置中打开，现有单帧 raw ACT + mechanical
assist 参考路径没有被改写，也没有部署到真机。

回退条件非常明确：删除或关闭 `temporal_input.enabled`，就回到原来的单帧输入；继续
使用已有的 G42/A checkpoint，则模型结构和输入形状也回到原来的路径。HDF5 原文件、
action scale、机械死区数值、E52 gate 和 previous-command policy 输入都没有修改。

# [Execution target G43/H2: 实现边界与输入契约]

- `testbed/testbed/data/causal_visual_history.py` 负责因果窗口的缓存、启动时首帧填充、
  时间戳检查和 reset；它不做动作决策、死区处理或 gate。
- 数据集在读取第 `t` 个样本时只取 `[max(0,t-3), ..., t]`，启动阶段重复第 0 帧，
  不读取未来帧。
- adapter 在 episode/reset 或时间戳断点清空缓存；运行时输入变成
  `(batch, history=4, cameras, channels, height, width)`。没有可靠时间戳时只使用同一
  观测调用内的严格顺序回退，并记录诊断。
- ACT 的 temporal mixer 以“最新帧恒等映射”初始化。旧单帧 checkpoint 加载时只缺少
  temporal mixer 的两个参数，因此关闭 temporal input 时旧路径保持兼容。
- 所有默认配置仍是 `enabled: false`；G43 的独立配置是
  `testbed/testbed/configs/act_real_gmsl_eye2_g43_temporal_history4_effective_action_2000.yaml`。

# [Execution target G43/H3: 训练与数据封存]

训练使用现有 24 个正式 episode、G37 固定 train/validation split、2000 epochs，从已
验证的 G42 effective-action-style checkpoint 初始化。没有使用 held-out `episode_105..109`，
也没有修改源 HDF5 或数据协议。

主要产物目录：

`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260713/g43_temporal_history4_effective_action_2000/`

- best checkpoint SHA256：`37396461a0fd67b355cb6f30d1552991d57770fe085cc7028cf70457a87c04b4`
- resolved config SHA256：`32aaac23dfc23347c818bcf9387717ef1eee0fb51dcfd10cf4bbb72d3c8f9b77`
- run metadata SHA256：`0cb3550f04818f70e5a1f69aa551f679b3574ff4e0057dd3e495b0b3141821a6`
- 最佳 validation loss：`0.12359288055449724`（epoch 930）

# [Execution target G43/H4: 完整离线测试结果]

测试包含 24 episode 的开放环动作审计、5 个 validation episode 的递归 state-hold
死锁/隐藏动作检查、tail/release 安全审计和已有 execution-response sidecar 的因果
响应审计。held-out 没有读取。

| 指标 | G43 短时序 4 帧 | G42 H5 参考 | G38 A 参考 |
|---|---:|---:|---:|
| 开放环整体 MAE | 0.04944 | 0.0471 | 0.05158 |
| state-hold raw | 40/48 | 41/48 | 40/48 |
| state-hold + assist | 44/48 | 43/48 | **46/48** |
| assist hidden | 1 | 2 | 1 |
| startup | 5/5 | 5/5 | 5/5 |

G43 的开放环方向一致性在长期有效动作段为 `98.10%`，extra/wrong 为 `0.42%`；但全
episode 统计中 extra/wrong 为 `15.97%`。这说明短时序确实帮助了已经开始的动作保持方向，
却在一些无动作或切换状态产生了过早/额外有效输出。输出越界和非有限值均为 0，单 tick
最多同时有效 2 个轴。

state-hold + assist 的 4 个死锁 anchor 为：episode 94 的 boom+，episode 74 的两个
boom+，以及 episode 92 的 bucket-；仍有 1 个 hidden anchor。因此完整 promotion gate
（assist 至少 46/48 且 hidden=0）未通过。

已有响应 sidecar 在正式 train/validation 选集上共 240 个 command onset，236 个观察到
同向响应、4 个为启发式 stalled candidate；由于它不含 policy intent、人工纠正和真实
失败标签，本轮没有拟合或启用 retry 策略。gohome 仍无法从现有正式 HDF5 估计，因为没有
完整 handoff 标签。

# [Execution target G43/H5: 结论与回退决定]

短时序输入不是无效方向：相对于 G42 H5，assist state-hold 从 43/48 提到 44/48，
hidden 从 2 降到 1，长有效动作段的方向保持很好。但它没有超过当前最强参考 G38 A 的
46/48，而且没有把 hidden 降到 0；全 episode 的额外有效输出也明显偏高。

因此本轮候选判定为“离线有信号，但不晋级、不部署”。保留 G38 A/raw ACT + mechanical
assist 作为回退和比较基线；G43 只作为独立离线实验产物。下一步若继续，应针对启动前的
“何时坚定越过死区”增加显式启动/持续状态监督或短时视觉差分，并继续使用同一完整
state-hold gate；不应把这次 temporal mixer 直接接入现场默认配置。

# [Execution target G43/H6: 可复核文件]

- 代码与测试：`testbed/testbed/data/causal_visual_history.py`、
  `testbed/testbed/policies/act/detr/models/detr_vae.py`、
  `testbed/tests/test_causal_visual_history.py`、`testbed/tests/test_act_temporal_model.py`、
  `testbed/tests/test_temporal_input_wiring.py`
- 完整报告：
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260713/g43_temporal_history4_effective_action_2000/complete_offline_report/complete_offline_report.json`
  SHA256：`2063928f38fa36ebeb559da1a683efa5c7f1c1f8f9e59f65430a1b700f4c2bb5`
- 开放环 collection summary SHA256：`b1b536f42fa95746aa3af0099bfe7c1b2ef5418679b2c42a5acd67168e3bff13`
- state-hold summary SHA256：`63400aefa8358772cfeb1ecf47d6b48ebaff58ebf37a07fa46ef45312ff315fe`
- execution monitor JSON SHA256：`e4eff447206b2c1fd9d0d6c5397eaef936ba829f78908837a114c77e43c26b48`
