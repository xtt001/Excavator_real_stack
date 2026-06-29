# Policy 到 Go-home 的交接思路

本文记录 2026-06-16 针对 policy 尾段乱动问题的当前判断和后续设计方向。它不是正式实现计划，
只用于固定问题定义和避免后续继续沿着错误假设推进。

## 当前结论

1. 不应该要求模型凭空学会停下。

   当前录制流程里，人工会把机器回到 dig/near-home 区域，然后按 gohome 让 PD 控制器回原位。
   训练数据已经排除了 gohome，因此模型没有、也不应该学习 gohome。尾段数据里人工仍可能在做回位
   微调，所以模型继续输出回位动作符合监督学习目标。

2. 问题本质是缺少 policy-to-gohome 的明确交接边界。

   当前 runtime 会继续采用 policy 输出，直到人为切换或停止；如果机器已经回到 gohome 可接管区域，
   继续让 policy 控制就会把“回位尾巴”延长成真机有效动作。

3. 加 qvel 不能单独解决。

   qpos+qvel ACT full2000 已经和 baseline 同等训练预算，但在 live-like 66-70 上 end80
   extra/wrong 从 qpos-only baseline 的 `18.25%` 增加到 `31.5%`。qvel 能说明机器还在运动，
   但不能说明当前动作应该属于 policy 任务内行为，还是应该交给 gohome 的收尾行为。

4. Deadzone assist 只能放大已有意图，不能解决交接语义。

   assist 默认关闭。历史回测显示它会把尾段本来低幅或错误时机的动作抬过死区，因此不应该用 assist
   作为尾段问题的修复手段。

## 推荐方向

做一个轻量的 `PolicyToGoHomeHandoff` 判据，而不是把整个流程写成复杂状态机。

核心思想：

```text
policy_action = model(obs)
handoff_score = detector(qpos, qvel, policy_action)

if handoff_score 连续稳定超过阈值:
    停止采用 policy_action
    启动现有 gohome controller
else:
    继续采用 policy_action
```

第一版只服务当前单一场景，可以从最小条件开始：

- swing 已经回到 dig/near-home 可接管区域；
- swing qvel 不超过安全阈值，避免高速扫过时误触发；
- 条件连续满足一个短 dwell 时间，例如 `0.3-0.5s`；
- 触发后带 hysteresis，不在边界附近反复进出；
- 触发后走 runtime 内部 gohome request 路径，而不是依赖物理 button3 输入。

这个设计仍然有少量状态，但状态只用于防抖和一次性触发，不承担完整任务阶段推理。

## 为什么不先做 terminal classifier

当前没有干净的 terminal label。尾段数据包含人工回位微调，不能简单标成 stop。训练分类器容易把同一个
语义混杂问题搬到另一个模型里。相比之下，先用 qpos/qvel 的可解释几何判据做 handoff，更容易离线验证、
现场调参和安全回退。

## 离线验证口径

实现前应先做只读离线验证：

1. 用 HDF5 中的 `qpos/qvel/action` 回放 detector，记录每集触发 step。
2. 检查触发点是否落在专家主要任务动作完成之后、gohome 之前。
3. 对比触发后截断 policy 输出的 end80 deadzone 指标：
   - live-like 66-70；
   - all31；
   - 特别检查 episode 66/67，因为它们当前尾段 extra/wrong 最明显。
4. 输出 per-episode CSV，包含 `handoff_score`、`triggered`、`trigger_reason`、`qpos`、`qvel`、
   `expert_action` 和 `policy_action`。

## Runtime 诊断字段

现场日志至少应新增：

- `policy_handoff_enabled`
- `policy_handoff_score`
- `policy_handoff_triggered`
- `policy_handoff_reason`
- `policy_handoff_dwell_steps`
- `policy_handoff_qpos_error`
- `policy_handoff_qvel`
- `policy_handoff_policy_action`

这些字段必须和现有 `policy_action`、`policy_assisted_action`、`safe_action`、`commanded_action` 同时保留，
否则后续无法区分模型输出、交接裁决和实际下发动作。

## 潜在风险

1. 过早交接：如果 swing 判据太宽，policy 还没完成回位就切到 gohome。
2. 高速穿越误触发：只看 qpos 不看 qvel，会在 swing 快速扫过边界时误触发。
3. 单场景过拟合：当前可以先只看 swing，但后续换 dig/dump 区域后需要用采样区域或多轴约束替代单阈值。
4. gohome 启动区域过宽：当前配置里 `near_tolerance_rad` 临时很大，真正部署前需要恢复为现场标定区域。
5. 日志不可追溯：如果只下发 gohome、不记录 handoff 判据，现场失败后无法判断是模型、判据还是 gohome 控制器的问题。
6. 交接后反复切换：必须有一次性 latch 或 hysteresis，触发 gohome 后不应自动恢复 policy。

## 当前建议

先不要改人工录制流程，也不要启用 deadzone assist。下一步应做离线 handoff detector 回放，确认
“swing 回到 near-home 并稳定一段时间”是否能覆盖当前尾段乱动，同时不截断主要任务动作。
