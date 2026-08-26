# condition action loss weight 审计（2026-08-26）

## 结论

当前没有任何一个 weight 可以晋升为真机候选。

需要特别修正一个验收口径：ctx04 的共同动作是先向约 1.6–1.9 rad 的卸料区摆动，再统一负向回转；A/B 的区别主要是回转停止位置和释放时刻。因此，旧版 same-state `+1/-1` swing 差值的正负不能当作目标方向验收。weight=1/2/5 虽然能把分类头训练到 100%，仍不能据此证明模型会在正确位置释放回转动作。

这次审计也确认了此前流程错误：历史的 1/5 长训使用 `anchor_only`，当前补做的 weight=0 以及本轮 sweep 使用 `all_active_steps`。它们不是同一个 loss 计算模式，历史 1/5 不能作为严格的 weight-only 对照。

## 固定协议

- 数据：`session_rt_20260822_ctx04_v2`，同一 `train_ready_manifest.json` 和同一 source-block split。
- 模型：当前 qpos + `real_transition_condition_v1` ACT，旧 qpos checkpoint warm-start；seed=0，batch=4，lr、deadzone/state-hold、goal-effect 和 4 路图像均不变。
- 条件损失模式：`all_active_steps`，记录条件和同一观测翻转条件都参与 deterministic action-class CE。
- 候选：`0, 0.5, 1, 2, 5`。0 是无条件损失负对照；0.5/1/2 覆盖低、近似同量级和强约束；5 是历史强约束对照。
- 每个候选 300 epoch；best checkpoint 只由 validation 总 loss 选择。随后对 15 个 validation episode 做 8 次重复条件 CE 审计、same-state `+1/-1` query-0 作用审计和死区/数据支持门控 mock 闭环。

## 结果

| weight | best epoch / train val loss | 条件 CE（未加权） | 条件准确率 | 重复审计总 loss | 旧 query-0 差值正数（仅诊断） | mock 完成 | mock 有效 swing |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 240 / 0.331145 | 0.718726 | 50.0% | 0.415228 | 7/7（差值约 0.0005，不能视为学到条件） | 2/15 | 2/15 |
| 0.5 | 240 / 0.682331 | 0.692817 | 51.7% | 0.746865 | 7/7（同样接近零差值） | 2/15 | 3/15 |
| 1 | 280 / 0.360459 | 0.003784 | 100% | 0.425078 | 1/7 | 1/15 | 1/15 |
| 2 | 240 / 0.305688 | 0.002190 | 100% | 0.417232 | 1/7 | 2/15 | 3/15 |
| 5 | 280 / 0.353527 | 0.000861 | 100% | 0.459445 | 0/7 | 0/15 | 0/15 |

旧 query-0 差值来自初始观测下的 `action(condition=+1) - action(condition=-1)`；7 个样本是 validation 中 `A->B` 或 `B->A` 的 episode。它只证明两个输入分支的连续输出是否不同，不能证明目标位置方向。mock 的完成/有效 swing 受 train-only qpos 支持门控约束，是数据标定的离线诊断，不是液压或真机效果证明。

另外，条件翻转本身的 swing 差值仍远小于直接策略死区（正向 0.661、负向 0.721）：weight=0/0.5 的中位数约 0.0005，weight=1/2/5 的绝对最大值约 0.016/0.016/0.014。高 weight 产生了可分类的 action 差异，却没有产生可执行的条件 swing 差异。

ctx04 的专家轨迹显示，回转段对 A/B 的正确判据应是“在目标侧区域释放负向回转并稳定”，而不是要求 A 负向、B 正向。训练数据中 A 目标回转段中位约 218 个 20 Hz step、有效负向命令约 146 个，B 目标分别约 180 step、109 个；目标信息主要体现在释放时刻和最终 ready 区域。新的 planner 开环参考回放已把这个几何判据和 3/4/5-cycle 连续 planner 生命周期分开实现。

## 为什么 1 和 5 不能直接拍板

本实现的总目标可写成：

```text
L = L_BC + 10 L_KL + L_deadzone + L_goal_effect
    + weight * CE(condition_action_head)
```

`condition_action_head` 的 CE 是从整段 action chunk 的 deterministic ACT proposal 做二分类；它不是 swing 符号、目标端点距离或死区穿越的直接奖励。当前 `condition_adherence_loss.directional_enabled=false`，所以 recorded-crossing、counterfactual-violation、contrast 三项实际为 0。weight 只放大 CE，不能自动变成“动作必须朝目标端点走”的强化约束。

在本轮共同初始化的 epoch 0 validation 中，未加权 CE 约为 0.738，其他项合计约 0.925。因而 weight=1 的 CE 约与其余目标同量级，weight=5 的 CE 约 3.69，已经约为其余目标的 4 倍。5 是强分类压力，不是中性默认值。

审计产物：

- [all-active weight sweep JSON](/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_ctx04_b1_condition_weight_sweep_allactive_pilot300_v1.json)
- [condition sensitivity JSON](/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_ctx04_b1_condition_weight_sensitivity_allactive_v1.json)
- [planner open-loop validation replay](/data/pingfan/Excavator_real_stack_data/runs/planner_open_loop_replay_allactive_validation_v1.json)
- [planner open-loop locked replay](/data/pingfan/Excavator_real_stack_data/runs/planner_open_loop_replay_locked_v1.json)
- 各候选目录下的 `mock_closed_loop_eval_allactive_validation.json`

## 后续决策

不再继续盲扫 0.25、3、10。现有 0–5 已经显示：更大的 weight 能让分类头更快、更低 CE，却不能保证连续动作在正确目标侧释放。下一步应固定一个中等候选（暂记 weight=2，仅作数值候选，不作发布候选），用 planner 开环参考回放检查共同卸料段、目标侧回转释放和连续组合；在该直接动作门通过前，不进入真机部署。
