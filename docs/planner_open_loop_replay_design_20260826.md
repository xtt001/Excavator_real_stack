# Planner 驱动开环参考回放（2026-08-26）

## 结论

当前模型不能证明已经具备“按 planner 的 A/B 目标完成一铲并在目标侧释放回转”的能力。

在锁定的连续 planner 回放中，低 weight（0/0.5）与专家动作的一致性最好，但同观测目标翻转几乎不改变目标侧释放动作；weight=1/2/5 的条件分类头可以变好，连续挖掘/回转动作却逐步退化；加入 qvel 的 2000 epoch 候选也没有改善。结果只能支持“planner 生命周期和动作生成链路可运行”，不能支持“真机动作已经可执行”。

## 从数据冻结的真实运行范围

本批数据的 HDF5 是 20 Hz 派生记录，原始现场控制链是 50 Hz。训练和离线 policy replay 使用 20 Hz；每个 cycle 的硬停止范围按现场合同为 60 s（1200 个 20 Hz step），train cycle 长度 P95 为 902 step（约 45.1 s）。

从 60 个 train cycle 得到的范围：

- ready A：`[-0.378786, -0.093129] rad`；ready B：`[0.111158, 0.392764] rad`；典型 q05–q95 分别为 `[-0.336390, -0.123110]` 和 `[0.143728, 0.362419]`。
- ready 稳定窗：连续 10 个 20 Hz 样本（0.5 s）`abs(swing_qvel) <= 0.015 rad/s`。
- 直接策略死区：swing `+0.661/-0.721`，boom `+0.259/-0.357`，stick `+0.5/-0.5`，bucket `+0.408/-0.508`。
- 每个专家 cycle 先进入共同的 swing-to-dump 中间段；swing apex 的 train q05/median/q95 为 `1.613/1.723/1.857 rad`。ready 安全范围只用于边界，不能作为全程工作范围。
- 回转段对 A/B 都主要使用负向 swing 命令。A 目标回转段中位约 218 step、有效负向命令约 146 个；B 目标约 180 step、109 个。A/B 的主要区别是负向回转何时释放、最后停在哪个 ready 区域，不是 A 负向、B 正向。

## 实验逻辑

`tb-planner-open-loop-replay` 按真实 runtime 的因果顺序执行：

1. 读取一个完整 held-out source run 的初始 ready 侧和变长目标序列（3/4/5 cycle）。
2. planner 在每个 cycle 开始前提交一个 A/B goal；ACT 输入只附加当前 `real_transition_condition_v1`，不读取后续目标。
3. policy 按连续模式保持 ACT/temporal aggregation 状态跨 goal；只在 source run 起点 reset。
4. 每个 20 Hz 记录观测输入四路图像、qpos、qvel，经过 action guard，分别统计 `policy_action`、`safe_action` 和专家 action 的死区/阶段/误差。
5. 用记录中的 apex、target-entry 和 ready 稳定窗划分共同挖掘段、回转段和目标释放段；在释放段对同一观测做 A/B 条件翻转 query-0 干预。
6. 参考 ready 边界只用于推进 planner 生命周期。模型 action 不反写 qpos/qvel，因此结果是开环动作能力证据，不是假装完成了物理闭环。

之前的线性 qvel 积分器没有通过“专家 action 回放复现”检查，位置误差会累积到数 rad；它没有被用于下面的模型判定。数据积分器仍可作为后续独立研究对象，但必须先通过专家复现门。

## Validation 结果

Validation 有 3 条完整 source run、13 个 reference cycle，所有候选的 planner 生命周期均能按记录边界走完。专家参考的平均阶段基线为：挖掘段正向有效率约 0.241，回转段负向有效率约 0.659，目标释放段 idle 率约 0.940。

| 模型 | action MAE | sign agreement | 挖掘正向有效率 | 回转负向有效率 | 释放 idle 率 | 翻转后目标几何命中率 |
|---|---:|---:|---:|---:|---:|---:|
| weight=0 | 0.099 | 0.951 | 0.250 | 0.693 | 0.868 | 0.062 |
| weight=0.5 | 0.094 | 0.948 | 0.261 | 0.688 | 0.852 | 0.082 |
| weight=1 | 0.131 | 0.939 | 0.240 | 0.637 | 0.945 | 0.077 |
| weight=2 | 0.146 | 0.831 | 0.130 | 0.455 | 0.926 | 0.091 |
| weight=5 | 0.208 | 0.802 | 0.131 | 0.395 | 0.973 | 0.120 |
| qpos+qvel, weight=5 | 0.188 | 0.796 | 0.089 | 0.351 | 0.983 | 0.038 |

这里的“翻转后目标几何命中率”不是分类头准确率，而是同一释放观测把目标换成另一侧后，query-0 action 是否按目标几何产生应有的继续回转/停止动作。所有候选都很低，说明条件输入没有稳定改变可执行的目标释放行为。

## Locked test 结果

Locked block 计划 16 个 cycle，实际有 13 个可回放 cycle，其中 2 个 reference cycle 本身缺少完整 apex/ready 边界，按数据原样记为不完整；它们不是模型失败，不能被静默删除。其余 11 个 reference cycle 的动作统计如下：

| 模型 | action MAE | sign agreement | 挖掘正向有效率 | 回转负向有效率 | 释放 idle 率 | 翻转后目标几何命中率 |
|---|---:|---:|---:|---:|---:|---:|
| weight=0 | 0.108 | 0.913 | 0.257 | 0.682 | 0.709 | 0.177 |
| weight=0.5 | 0.107 | 0.915 | 0.277 | 0.668 | 0.731 | 0.161 |
| weight=1 | 0.138 | 0.910 | 0.239 | 0.600 | 0.860 | 0.156 |
| weight=2 | 0.151 | 0.824 | 0.141 | 0.438 | 0.864 | 0.156 |
| weight=5 | 0.203 | 0.799 | 0.152 | 0.400 | 0.908 | 0.193 |
| qpos+qvel, weight=5 | 0.194 | 0.791 | 0.089 | 0.318 | 0.909 | 0.120 |

Locked 结果没有出现一个同时满足“共同挖掘段、负向回转段、目标释放干预”三项的模型。低 weight 的 sign agreement 较高，主要说明它更像专家回放；它没有通过 condition flip，因此不能视为 A/B planner-following。

## 证据边界

- planner、goal epoch、变长顺序和跨 cycle policy 状态已被真实 source-run 序列覆盖；这部分说明脚本式 planner 接口可接通。
- policy action 在 recorded qpos/qvel/image 观测上的有限性、死区判定和动作层级均已覆盖；这部分没有发现数值 NaN 或 guard 链断裂。
- 该实验没有把模型 action 写回真实状态，所以不能证明液压响应、土量、卸料效果或真机最终落点。
- 当前最重要的失败不是 planner 顺序，而是目标条件没有稳定改变“回转何时释放”。因此继续调 weight 不能替代直接的目标侧释放约束和同观测 A/B 配对监督。

## 产物

- [validation replay JSON](/data/pingfan/Excavator_real_stack_data/runs/planner_open_loop_replay_allactive_validation_v1.json)
- [locked replay JSON](/data/pingfan/Excavator_real_stack_data/runs/planner_open_loop_replay_locked_v1.json)
- [replay CLI](/home/pingfan/Excavator_real_stack/testbed/testbed/cli/planner_open_loop_replay.py)
- [train-only calibration](/home/pingfan/Excavator_real_stack/testbed/testbed/data/open_loop_experiment.py)

SHA-256：validation `634c0e42ac1ec49b30f869f6a3dfbdfa83de48f8af326573f901d035d2782b4a`；locked `b80f17c4d1b7c86360457038af7fa04b64dcc90019c4d32d9f297c41d95cfdbe`；qvel validation `450574cfa91b26da0202f4c6b2fad050b5a406967de92f359c3dc19a9a9503c6`。
