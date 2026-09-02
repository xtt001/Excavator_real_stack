# Real Transition task-state-v2 自动推进运行时

日期：2026-09-02

## 结论

task-state-v2 的正常运行不再要求操作者逐铲按按钮。任务脚本继续定义初始区域、每个
cycle 的下一目标和执行顺序。运行时根据当前 cycle 内已经发生的物理运动和策略动作，
自动提交 `WORK_COMPLETE` 和 `RETURN_COMMITTED`。

自动判断使用因果历史，不读取未来帧，也不把单一 swing 位移当作整铲完成。缺少大臂和
挖斗运动、正向 excursion、有效卸料动作或动作收稳证据时，状态保持不变，最终由现有
review/timeout 机制停止。卡死状态不会被当作完成。

## 数据合同

自动推进条件在 90 条完整 cycle 上冻结，其中 train 60 条、validation 15 条、locked
test 15 条。数据来源、episode identity、split 和 SHA-256 沿用冻结的 task-state 与
work-context manifest。

专家数据得到以下结果：

| 检查 | 结果 |
|---|---:|
| work-complete 前 boom qpos 位移不小于 0.05 rad | 90/90 |
| work-complete 前 bucket qpos 位移不小于 0.05 rad | 90/90 |
| 正向 excursion 后第一段有效正向 bucket 动作就是 hindsight work 段 | 90/90 |
| 有效正向 bucket 段最短长度 | 24 帧 |
| 两帧 bucket 释放窗口早于有效回程 | 90/90 |
| work-complete 后两帧全轴动作位于机械死区内，且早于有效回程 | 90/90 |
| 最后一个条件到有效回程的最小余量 | 7 帧 |

冻结运行参数为：

- boom 和 bucket 都必须相对 cycle 起点产生至少 0.05 rad 位移；
- swing 必须通过既有正方向 excursion 合同；
- bucket 正向动作必须超过机械阈值 0.408，并连续至少 5 帧；
- 随后连续 2 帧低于正向 bucket 阈值，生成待提交的 `WORK_COMPLETE`；
- `WORK_COMPLETE` 生效后，连续 2 帧所有策略动作位于对应机械死区内，生成待提交的
  `RETURN_COMMITTED`。

5 帧 bucket 门槛低于专家数据最短的 24 帧，不会要求数据中不存在的持续时间。机械阈值
直接来自既有 direct-policy-output deadzone contract。

冻结合同：

`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/auto_progress_contract_v1/task_state_auto_progress_contract.json`

SHA-256：`abae0df49228e1c679d25dfeecba780bab3551ebe6bcfef02fe1ef1d9421565e`

## 自动状态机

每个 goal commit 后，状态从 WORK 开始：

```text
WORK
  ├─ boom 和 bucket 有真实 qpos 位移
  ├─ 正方向 swing excursion 已确认
  ├─ 有效正向 bucket 动作连续至少 5 帧
  └─ bucket 动作释放连续 2 帧
       ↓
WORK_COMPLETE
  └─ 全轴策略动作连续 2 帧位于机械死区内
       ↓
RETURN_COMMITTED
  └─ 下一目标侧公开给 ACT
       ↓
目标侧 ready 稳定窗口通过
       ↓
脚本自动提交下一 cycle
```

事件先在当前动作之后进入 pending，下一次策略推理前才修改 token。这样当前动作仍与产生
它的旧 token 对齐。token 改变时同时 reset ACT，避免旧 chunk 和时序聚合跨阶段泄漏。

任务脚本仍是唯一的目标顺序来源。自动状态机只判断当前 cycle 的进度，不生成目标，不
替代 ACT 输出连续动作。

## 候选模型交互 replay

冻结自动合同确定后，在 30 条 source-disjoint held-out cycle 上运行候选 checkpoint。
自动状态机由模型聚合动作驱动；qpos、qvel 和四路图像继续使用录制序列，因此这是
recorded-state open-loop replay，不是物理闭环。

| 指标 | 全部 30 条 | B→A 8 条 | 其他 22 条 |
|---|---:|---:|---:|
| 自动得到 WORK_COMPLETE | 100% | 100% | 100% |
| 自动得到 RETURN_COMMITTED | 100% | 100% | 100% |
| boom/bucket liveness 通过 | 100% | 100% | 100% |
| 有效 bucket work 通过 | 100% | 100% | 100% |
| commit 前机械有效负向 swing | 6.7% | 0% | 9.1% |
| commit 后出现机械有效负向 swing | 86.7% | 100% | 81.8% |
| 自动 commit 早于录制有效回程 | 96.7% | 100% | 95.5% |

B→A 的 8 条 held-out cycle 均在自动 commit 前保持无机械有效负向 swing，并在 commit
后产生有效负向回程。全部样本中的 2 条 commit 前负向动作来自其他 transition，与已授权
的 allow-2/29 候选边界一致。

replay 结果：

`/data/pingfan/Excavator_real_stack_data/runs/real_transition_v2_0_1_task_state_v2_v1/auto_progress_replay_v1/auto_progress_replay.json`

SHA-256：`6a61c5e17c7108473a4c68a1638cc7c67d9e557659b454f0844fecb770937cb3`

## 运行与安全边界

物理按钮 7 继续负责 ARM、退出模型和解除 script-stop latch。正常挖掘不再绑定 task-state
mark 按钮。脚本开始后，cycle 计划和阶段状态都由程序推进。

`shadow_zero` 仍是首次现场运行要求。日志必须显示 work liveness、excursion、bucket
effective、bucket release、action idle、WORK_COMPLETE 和 RETURN_COMMITTED 的有序链路。
缺少任一证据时不得进入 control。

本轮没有连接 Jetson、现场 TCP 或液压系统，没有进行真机运动。自动 replay 不能证明
策略动作会在物理闭环中产生相同 qpos/qvel、卸料效果或土方结果。候选仍保持
`OFFLINE_CANDIDATE_ONLY`，需要 shadow_zero 和受控真机验证。
