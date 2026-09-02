# Real Transition task-state-v2 离线实验

日期：2026-09-02

后续状态：本文保留训练和候选选择时的原始结论。文中“在线 owner 尚未实现”的边界已由
同日的自动推进运行时补齐，当前实现和新增 replay 结果见
`docs/real_transition_task_state_v2_automatic_runtime_20260902.md`。

## 结论

严格的原始冻结验收没有产生候选。原始专家能力复核后，用户明确允许 29 个 held-out 未 commit 窗口中偶发 2 条提前负向 swing。按这一结果后授权的新要求重新检查全部 26 个 checkpoint，选出 worst-query run 的 epoch 15 作为 `OFFLINE_CANDIDATE_ONLY`。它仍不得进入现场运动，必须先补齐 task-state runtime owner 并完成 shadow_zero。

实验确认了两点。第一，旧 accepted epoch 199 的 B→A 直接到 A 捷径可以在固定离线 probe 中稳定复现：8 条 source-disjoint held-out B→A cycle 中有 6 条出现捷径，比例为 75%。第二，把“当前侧、挖掘侧、挖掘完成、允许回程、下一目标侧”写成独立任务状态后，所有新模型在同一 8 条 B→A 上都把捷径降到了 0%。因此，原先只给最终目标的 condition 合同确实缺少当前动作语义。

捷径率为 0 仍不足以晋级。后续模型无法同时满足 B→A 停稳后启动有效工作，以及 `dig_complete=1` 且 `return_commit=0` 时不提前回程。最接近的模型将未 commit 防抢跑从 21/29 提高到 27/29，但部分 checkpoint 的 B→A 正确起步从 7/8 降到 5/8。继续增加 loss、margin 或训练长度已经表现出明确的语义竞争，所以实验停止，没有包装 checkpoint。

实验后对验收合同做了原始专家数据对照。该复核确认，零 qvel 活性、固定 token/qvel raw-chunk 差值以及其他 transition 75% 起步率缺少原始数据支持，不应作为硬门槛。移除这些门槛并按原始专家覆盖重新检查全部 26 个 v2-probe checkpoint 后，严格结论仍为无候选：26/26 均未达到至少 28/29 未 commit 窗口无负向 swing 的门槛。用户随后明确把容忍度改为最多 2/29，形成新的结果后授权合同。

## 现场问题

现场 B→A cycle 切换目标后，accepted epoch 199 在首步直接输出约 `-0.84` 至 `-0.88` 的 swing 动作，从 B 向 A 移动，跳过预期的右侧挖掘 excursion。正常 B→A 首步约为 `-0.0483`。日志已经把异常定位到原始 `policy_action`，排除了 landing、PD、安全层和 deadzone assist。

旧 condition 只表达最终目标侧。对于 B→A，它从 cycle 第一帧开始就告诉模型“目标是 A”，但没有明确说明当前应当在 B 工作、工作是否完成、何时允许回程。训练数据中的姿态和速度又与阶段高度相关，模型可以依靠相关性猜动作，不必真正服从任务指令。

## 离线复现

冻结 probe 使用 30 条 source-disjoint held-out cycle，其中 validation 15 条、locked test 15 条，B→A 共 8 条。每个动作窗口在评价前回放最多 19 帧真实观测，用于恢复 20-query ACT temporal aggregation。task-state 模型在两个事件位变化时 reset；旧模型只在 cycle goal commit 时 reset。

每条 cycle 检查以下窗口：

- cycle 起始的工作动作；
- 正向 swing excursion；
- bucket/tool 工作；
- 第一个任务状态边界；
- 机械有效回程；
- B→A 回程中 swing qpos 再次经过本 cycle B-ready swing 位置的窗口；
- 末尾保持。

probe 同时保存每步 raw action chunk、temporal aggregation 后的 policy action、机械死区方向、首次有效 swing、动作积分代理、MAE 和有效动作符号一致率。未来 qpos、图像和土方状态始终来自录制序列，模型动作不会改变下一帧，因此这是 state/history-conditioned open-loop replay，不是物理闭环。

旧 accepted epoch 199 在该 probe 上得到：

| 指标 | 结果 |
|---|---:|
| B→A 直接捷径率 | 6/8，75% |
| B→A 正确起步 | 2/8，25% |
| B→A 回程负向 swing | 8/8，100% |
| B→A 回程经过 B 时继续负向 | 8/8，100% |
| 全 held-out 聚合动作 MAE | 0.1286 |

旧模型在两条现场异常 hybrid 上均复现捷径，正常 hybrid 不出现捷径。三条 hybrid 缺少现场真实图像，其中 qpos 或 qvel 由文档值注入，因此只作非门槛诊断。

## 根因证据

### hindsight 阶段标签

90 条完整 cycle 均满足 `work_complete` 早于机械有效回程。88/90 满足 `work_complete <= return_commit <= effective return`。episode 63 和 70 分别出现 return commit 比 work complete 早 44 行和 3 行。新合同保留这两个独立事件位，没有把它们强制改成互斥三分类。

时间上，work complete 到 return commit 的中位数为 0.5 秒，commit 到机械有效回程的中位数为 0.6 秒，work complete 到机械有效回程为 1.2 秒。`work_complete` 不能被解释为“下一帧必须立刻反向”。

宽阶段的 qpos/qvel 可分性接近 100%，但 work-complete 相邻两帧的低维可分性只有 50% 至 67%，四相机 embedding 约为 50% 至 53%。这说明录制序列足以区分远离边界的 WORK、SETTLE、RETURN，但不能只靠单帧观测可靠决定精确事件时刻。hindsight token 需要由任务状态机提供，不能继续让网络从姿态猜。

### qvel 与任务状态

父模型 epoch 259 使用 qvel 后，在事实录制轨迹上已经把 B→A 捷径降到 0%，B→A 正确起步为 7/8，回程经过 B 时继续负向为 8/8。把 qvel 置零会改变 raw chunk；父模型平均变化约 0.052，新 task-state 模型约为 0.04 至 0.05。模型并非完全忽略 qvel。

第一版验收曾要求在同一 B-ready 姿态、同一图像、`qvel=0` 时，仅把 token 改为 RETURN 就必须立即负向回程。该输入组合与录制事实矛盾：停在 B 的样本属于准备工作，真实回程经过 B 时 swing qvel 非零。v1 结果被完整保留为失败记录，但不用于后续结论。v2 改为检查真实回程经过 B 的录制窗口，并把同观测 token/qvel 干预作为敏感性检查，不再给矛盾反事实指定动作标签。

### 当前数据是否可用

当前数据可用于：

- 稳定复现旧模型捷径；
- 追溯 work complete 和 return commit；
- 训练并验证新的任务状态输入；
- 检查事实 WORK、SETTLE、RETURN 和回程经过 B 的动作语义；
- 验证模型是否对 qvel 和 token 有响应。

当前数据不足以证明单一 ACT 网络能在所有边界状态同时保持工作活性并严格服从 commit。原因不是 hindsight 标签整体错误。更具体的限制是：精确边界在观测上弱可辨，RETURN 的姿态/qvel 与 WORK 强相关，未 commit 边界每条 cycle 只有一个起点，且不存在同一真实观测下由操作者分别执行 WORK/SETTLE/RETURN 的成对动作。

## 数据与训练修改

新输入为 13 维：`qpos4 + qvel4 + task_state5`。task state 定义为：

```text
[current_side_code,
 dig_target_code,
 dig_complete,
 return_commit,
 gated_next_target_code]
```

`gated_next_target_code` 在 `return_commit=0` 时固定为 0，只在允许回程后暴露 A/B。当前数据合同中 `current_side == dig_target`；未录制的独立组合会 fail closed。

源 HDF5 未被覆盖。新 sidecar 记录原 episode identity、split、source block/run、边界、候选起点和 episode SHA-256。采样按每 episode 四个等权 tier 进行：row 0 工作起步、完整 WORK body、第一个状态边界、两个事件均完成后的 RETURN body。动作 chunk 遇到任一 task-state 变化就截断，避免一个监督 chunk 同时包含两种 condition。

训练过程只在已有 state/visual residual 结构上训练低维动作头。语义 warm start 将父模型输入从 `qpos4 + old_condition2 + qvel4` 映射到新布局：qpos 和 qvel 按字段复制，旧 target 只映射到 gated next target，恒为 1 的 goal-active 列折叠进 bias，新事件位从零权重开始。视觉 residual 保持冻结。

固定预算实验如下：

| 实验 | 修改 | best epoch | 结果 |
|---|---|---:|---|
| task-state BC | 新低维头，普通 ACT L1 + 既有 deadzone loss | 80/100 | 捷径 0；起步和任务边界未同时通过 |
| semantic warm | 按字段复用父模型 qpos/qvel 低维头 | 80/100 | epoch 0 只差未 commit 防抢跑；后期活性下降 |
| mean guard | 未 commit 状态禁止越过负向 swing 死区，weight 1.0 | 5 | 最接近保留性结果；防抢跑 24/29 |
| margin guard | 根据最大越界 0.0484 固定 margin 0.06 | 10 | 防抢跑 27/29；B→A 起步降到 5/8 |
| worst-query guard | loss 从全 query 平均改为每样本最坏 query | 10 | 防抢跑仍为 27/29；无合格交点 |

没有进行 loss 权重搜索。

## 冻结验收

最终 v2 probe 和 v5 acceptance 在最后一轮训练前冻结。核心门槛包括：

- 8 条 B→A 直接捷径率为 0；
- B→A 正确起步和非 swing 工作活性至少 6/8；
- B→A 正向 excursion 至少 6/8；
- B→A 回程与回程经过 B 的继续负向率至少 7/8；
- 29 条正常 `dig_complete=1, return_commit=0` cycle 中至少 95% 不出现机械有效负向 swing，即至少 28/29；
- 其他 transition 的起步、excursion、tool、return 和有序动作代理保留；
- B-ready qvel 置零时至少 4/8 保持有效工作，且不出现直接回 A；
- token 与 qvel 干预均需让 raw chunk 产生可测变化；
- held-out 聚合动作 MAE 不超过父模型的 1.15 倍，有效符号一致率不低于父模型减 0.05。

tail idle 不作为门槛。原因是加入 qvel 后，强制要求 `qvel -> 0` 前后行为不变没有数据合同依据。现场异常 hybrid 也不作为门槛。

### 原始专家能力复核

原始数据不包含“return commit 时全关节完全停稳”合同。90 个 commit 帧中，83/90 的 action 在机械死区内，61/90 满足官方 swing qvel ready 阈值，只有 36/90 满足实验性四轴 qvel 稳定阈值；即使把近零定义为每轴 `|qvel| <= 0.001 rad/s`，也为 0/90。官方 ready contract 只约束 swing qvel，其他三轴 qvel 是 record-only unbounded。

88 个正常顺序 cycle 的 work-complete 到 commit 区间中，88/88 没有机械有效负向 swing，但只有 64/88 整段四轴 action 都在死区内，24/88 仍有机械有效工具动作，0/88 整段四轴 qvel 均稳定。因此，数据支持的语义是“commit 前不能启动有效负向回程”，不支持“commit 前后整机必须静止”。

held-out 专家覆盖为：B→A 起步有效工作 7/8，正向 excursion、bucket、回程和回程经过 B 均为 8/8；其他 transition 起步有效工作只有 15/22。历史 v5 acceptance 对其他 transition 要求至少 75%，相当于 17/22，高于原始专家覆盖。全 qvel 置零、token raw-chunk 最小差值和 qvel raw-chunk 最小差值也没有事实反事实标签，已降为诊断项。

按专家数据重新冻结的回溯验收保留以下事实门槛：B→A 起步至少 6/8、各主要动作段保留、其他 transition 起步至少 14/22，以及 29 个 held-out 正常未 commit 区间中至少 28/29 不出现机械有效负向 swing。全部 26 个已有 v2-probe checkpoint 都在最后一项失败，最高为 27/29。因此，无候选结论不再依赖零 qvel 或其他无数据支持的要求。

## 结果

代表性结果如下：

| 模型 | B→A 捷径 | B→A 正确起步 | 未 commit 不回程 | B→A 回程经过 B | 全 held-out MAE | 判定 |
|---|---:|---:|---:|---:|---:|---|
| accepted epoch 199 | 6/8 | 2/8 | 23/29 | 8/8 | 0.1286 | 暴露现场同类捷径 |
| parent epoch 259 | 0/8 | 7/8 | 25/29 | 8/8 | 0.1212 | 无 task-state authority |
| semantic warm epoch 0 | 0/8 | 7/8 | 21/29 | 8/8 | 0.1244 | 仅未 commit 门槛失败 |
| mean-guard best epoch 5 | 0/8 | 7/8 | 24/29 | 8/8 | 0.1225 | 仅未 commit 门槛失败 |
| margin-guard best epoch 10 | 0/8 | 5/8 | 27/29 | 8/8 | 0.1218 | 防抢跑与起步活性竞争 |
| worst-query best epoch 10 | 0/8 | 5/8 | 27/29 | 8/8 | 0.1219 | 无进一步改善 |
| worst-query last epoch 15 | 0/8 | 6/8 | 27/29 | 8/8 | 0.1209 | allow-2/29 离线候选 |

五版冻结 acceptance 的 `selected_candidate` 均为 `null`。episode 52 和 59 是后两种 guard 下仍持续出现未 commit 负向 swing 的两条 locked-test cycle。它们的专家未 commit 窗口都没有机械有效负向 swing，因此失败不能归因于验收标签要求了不存在的动作。

用户授权最多 2/29 偶发后，重新检查 v2 到 v5 的全部 26 个 checkpoint，共 4 个通过。冻结排序选择 `acceptance_v5:last_epoch15`，而不是训练 loss best。其 checkpoint SHA-256 为 `e57bd59f07650f674f58eb9dfdaae2c06ead22b903922039cb2e6400daacaa4b`。这一晋级来自运行要求变更，不代表模型达到了专家的 29/29，也不回写或删除此前严格失败结果。

结果图显示：任务状态把 75% 捷径降为 0%；guard 能提高未 commit 防抢跑；继续提高这一项会降低 B→A 起步活性。平均 MAE 持续改善并没有解决这个冲突。

## 尚未验证的真机边界

本轮没有实现 task-state 的在线 owner。部署前仍需由 planner/runtime 明确定义并记录 `current_side`、`dig_target`、`dig_complete`、`return_commit`、`next_target`，在状态变化时 reset policy，并验证 token 与日志逐步一致。

本轮没有连接 slave Jetson、`192.168.100.1` 或现场 TCP 链路，没有进行真机运动。离线结果不能证明液压响应、土方效果、相机时序、物理 excursion 或闭环回程。

已生成离线候选包，但 task-state 在线 owner 尚未实现，因此当前只能作为 shadow_zero 准备材料。若继续解决，优先级如下：

1. 停止继续堆神经网络 loss。先决定 `return_commit` 是否应成为系统级硬权限：在 `dig_complete=1, return_commit=0` 时，只禁止机械有效负向 swing，其他三个关节保持模型输出。该规则与 29/29 专家窗口一致，也不会直接压掉挖掘关节活性。这将形成新的系统合同，不能算作 raw 网络已经修复，必须另行冻结验收。
2. 如果仍坚持由 raw 网络独立满足全部门槛，应补充或构造成对的真实状态覆盖，重点是 episode 52/59 一类完成姿态、B-ready 零速度工作起步，以及相近 qpos 下非零 qvel 回程。仅再次切分同一连续轨迹或重复 hindsight 标签，不能增加这些动作条件的信息量。
3. 新数据仍需按 source episode/session 隔离，并在训练前冻结相同 probe。现场异常姿态继续保持非门槛诊断，等待实际图像和受控 shadow_zero 再判断。

## 产物

- task-state manifest：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/frozen_task_state_v2_manifest_v1.json`，SHA-256 `41bd0314190a29fe4eeaa47c8a789b3134f6b02ba031fc0916e60b08ef091429`
- 最终 probe：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/frozen_probe_manifest_v2.json`，SHA-256 `0e8b2449c9805a8ff83924c57f22d095469317c1236144728665bf6a24c7242d`
- 最终 acceptance contract：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/frozen_acceptance_contract_v5.json`，SHA-256 `880988ee9148647b6e5b9fc89750b6a54cca0b7530bff6e9333fc7240757c1e6`
- 最终 acceptance result：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/acceptance_v5/acceptance_result.json`，SHA-256 `3b04cdbb42057eba182081d46458d165532be69ef3397a7e6e96cbbad5317c5b`
- 汇总 JSON：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/consolidated_summary_v1/experiment_summary.json`
- checkpoint 表：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/consolidated_summary_v1/checkpoint_comparison.csv`
- 结果图：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/consolidated_summary_v1/behaviour_tradeoff.png`
- 原始专家 reference：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/expert_reference_v1/expert_reference.json`
- 专家对齐回溯验收：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/expert_aligned_redecision_v1/expert_aligned_redecision.json`
- allow-2/29 验收：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/expert_aligned_redecision_v2_allow2/expert_aligned_redecision.json`
- 离线候选包：`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/offline_candidate_allow2_v1`

代码回归：相关 ACT、dataset、policy 和 task-state 测试共 152 项通过，2 条既有 datetime deprecation warning。
