---
type: design-decision-log
project: Excavator Real Stack
version: v2.0.1-real-transition
status: accepted-for-current-prototype
created: 2026-08-12
updated: 2026-08-12
execution_authority: docs/v2_0_1_real_transition_data_definition_and_recording_plan.md
---

# v2.0.1 Real Transition 设计取舍记录

## 1. 文件用途

本文只记录 v2.0.1 方案形成过程中评估过的替代方案、否决理由和重新讨论条件。它不是现场任务卡、数据合同或执行依据。

数采、处理、训练和验收均以《v2.0.1 Real Transition 数据定义与数据录制计划》为准。两份文件发生冲突时，以主计划为准。

## 2. 已接受的设计

### D-01 使用以 home 为中线的左右二分类

当前原型把当天标定的 home swing 方向作为中线：左侧直接归为 A，右侧直接归为 B。A/B 是已示教 ready 区域，不是固定 qpos 点。home 附近保留只用于消除分类歧义的排除带。

qpos、qvel 支持范围和视觉参考用于判断样本是否落在本批数据支持内，不用于把师傅拉到某个固定角度。

### D-02 ACT 学四种 transition，脚本 planner 决定顺序

首批数据平衡覆盖 `A→A`、`A→B`、`B→A`、`B→B`。每条 run 连续完成四个 cycle，用来分别观察原子 transition 和多铲组合。

上层首版固定使用 `A-B-B-A-A` 和镜像 `B-A-A-B-B`。每条 run 自身都包含四种 transition 各一次。训练时按 ready-to-ready cycle 独立切片，ACT 输入不包含 planner 模板、run id、cycle index 或后续目标，因此 ACT 的职责是执行当前 transition，不是决定下一侧。

### D-03 condition 使用目标侧二分类

第一版 condition 为：

```text
[target_side_code, goal_active]
```

其中 A 为 `-1`，B 为 `+1`。当前侧由 qpos 和视觉状态表达，当前侧与目标侧共同确定四种 transition。这个表示只声称左右二分类，不声称任意角度控制。

### D-04 目标必须前瞻提交

每个目标由脚本 planner 在本 cycle 动作开始前提交并写入事件流。固定顺序对师傅是已知的，因此 condition 从 cycle 开始即有效。实际到达侧单独标注，不能反向覆盖程序目标。

### D-05 原始数据连续录制，训练数据离线切分

现场按完整四铲 run 保存原始数据；ready-to-ready cycle 在离线复核后生成。policy 和 temporal aggregation 只在 run 开始重置，cycle 边界只更新目标 lifecycle。

这一设计保留真实的多铲衔接，同时允许边界、condition 和 action mask 独立审计。

### D-06 qvel 先作为证据，不作为第一版 policy 输入

qvel 完整记录并用于 ready、QC 和稳定性判断。第一版 ACT 继续使用既有 qpos-only 低维主合同，只新增 condition。这样首轮实验中的主要新增变量保持为目标条件。

## 3. 当前未采用的方案

### X-01 地面坐标或沙箱分区作为目标

当前原型不建立地面笛卡尔坐标、沙箱格子或地面目标窗。此类方案需要额外解决机器与地面的配准、底盘位姿变化、扇形作业范围映射、目标可达性和土面变化后的目标有效性。

这些问题对未来的覆盖规划有价值，但不会帮助当前实验更直接地回答“ACT 是否使用 condition”以及“多铲 transition 能否组合”。在首批小数据中同时加入它们，会让失败同时可能来自目标配准、感知、可达性和策略学习，难以归因。

重新讨论条件：任务开始要求覆盖指定地面区域、避免重复挖掘、规划近远位置，或底盘移动后仍需维持统一场地任务时。

### X-02 直接使用 2×3 或更细网格

首批数据量不足以支撑多个格子和格子之间的 transition。格子越多，单格和单条边的样本越稀疏，模型也更容易依赖录制习惯或事后标签。出现误差时，很难区分数据不足、边界定义和 condition 失效。

重新讨论条件：左右二分类原型通过，并且后续数据预算能按目标和 transition 统计真实覆盖，而不是只统计总 cycle 数。

### X-03 用自由渐进挖掘直接验证 condition

自由挖掘适合高吞吐收集自然操作，但相邻两铲通常只发生小幅位置变化，当前状态、下一目标和师傅习惯高度相关。小批量实验中，即使模型忽略 condition，也可能复现相似轨迹并得到看似合理的结果。

它可以在原型通过后成为扩充数据的来源，前提是程序仍前瞻记录任务目标，并且离线数据能够形成足够清楚的目标干预和 transition 覆盖。

### X-04 使用固定目标角和连续角度误差

固定目标角会把“去左边或右边”收缩成“对准某个 qpos 点”，迫使现场围绕偏移量和支持带反复调参，也限制师傅在目标侧选择自然可挖位置。

当前只提供二分类 target side。连续 qpos 仍在 observation 中，ACT 可以根据当前机器状态和画面决定如何完成该侧 transition。

### X-05 首版加入视觉目标识别或视觉 Grounder

四路相机继续作为 ACT 的环境观察，视觉也用于左右区域参考和离线复核。首版不增加独立的视觉目标识别、地面目标检测或把目标提示叠加进训练图像。

当前左右分类可以由 home 参考和机器状态直接定义。新增视觉 Grounder 会同时引入感知标注、识别误差和遮挡问题，削弱 condition 实验的可归因性。

重新讨论条件：目标无法由机器状态唯一表达，或任务升级为外部地面目标、动态障碍和跨底盘位姿规划时。

### X-06 在 cycle 边界重置 ACT 或 temporal aggregation

cycle 间重置会把连续四铲变成四次独立演示，无法验证真实衔接。旧 chunk 可能包含卸料结束后的有效共享动作，因此当前先保留连续运行，并用 `goal_epoch`、chunk condition 和最终 commanded action 测量目标生效延迟。

只有证据显示旧 chunk 持续压制新目标时，才把聚合清除策略作为单独实验变量。

### X-07 把 A/B 定义为 home 两侧的固定偏移锚点

该方案虽然能制造清楚的角度差，但它把当前实验变成点目标控制。偏移量、对称误差和 endpoint 支持带会成为额外的成败来源，与当前只验证左右 transition 的目标不一致。

当前保留 home 作为分类中线，A/B 使用区域标签。排除带只处理 home 附近的歧义，不定义目标中心。

### X-08 让 ACT 学习或预测固定 planner 顺序

固定 P0/P1 只属于上层任务程序。ACT 训练文件按 cycle 独立切分，policy 输入明确排除模板 ID、cycle index 和后续目标。固定循环是否成功只证明 composition；condition 是否被使用，仍由 B0/B1/B2 和同一 observation 下切换 target side 的干预判断。

## 4. 决策边界

上述取舍只对 v2.0.1 小批量真机 transition 原型有效。原型通过后扩大目标范围时，应根据新增任务重新评估目标坐标系、planner 顺序、感知输入和数据覆盖，不能把本轮左右二分类成功外推为任意位置或跨场地能力。
