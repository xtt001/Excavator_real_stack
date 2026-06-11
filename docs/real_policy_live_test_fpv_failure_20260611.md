# 真机 policy live 测试失败记录（2026-06-11）

## 现象

当前 ACT 模型已能完成部分 `dig -> carry -> dump` 流程，但 live 测试稳定性不足：

- 共测试约 6 次，只有 2 次在启动阶段输出了足够大的动作。
- 多数失败表现为启动动作偏小，或 return 阶段 swing 方向正确但幅度不足。
- 一次成功挖/运/倒后，return 阶段 swing 输出约 `-0.57`，真实机器未能启动 swing。

相关日志示例：

```text
/media/mundane/EXTERNAL_USB/policy_control_tests/real_one_dig_v1_policy_test_20260610T094544.254078Z
/media/mundane/EXTERNAL_USB/policy_control_tests/real_one_dig_v1_policy_test_20260610T095831.182731Z
```

## 已排除

- 不是 CUDA 推理错误：该轮没有持续 `policy_error`。
- 不是安全 guard 或 action scale 把动作压小：`policy_action / safe_action / commanded_action` 基本一致。
- 不是单纯 qpos 跳变问题：bucket qpos 分支跳变问题在当前日志里没有复现。

## 主要判断

本次失败主要是 FPV 分布干扰，而不是控制链路问题。

证据：

- 固定同一组 return qpos，只换 live FPV，模型输出会随画面阶段明显变化。
- 在 live return FPV 上，关闭 temporal aggregation 后模型单步 swing 仍只有约 `-0.49 ~ -0.57`。
- 在训练集 return-like 帧上，同一模型能输出约 `-0.75 ~ -0.80` 的强负 swing。

所以当前模型虽然输入了 `qpos`，但阶段判断仍被 FPV 强烈主导。dump 区域土体状态、bucket/土堆位置、光照或遮挡变化，会让模型把动作幅度预测得偏小。

## 这次过错

1. 过早把 return 弱动作归因到 temporal aggregation 平滑和机器死区，证据不够完整。
2. live 前没有先做“固定 qpos、替换 FPV”的离线敏感性检查。
3. 当前训练数据对启动和 return 阶段的 FPV 变化覆盖不足，导致模型对土体/画面细节过敏。
4. 训练目标仍是 `FPV + qpos -> action`，但 qpos 没有足够约束模型按世界位置执行动作。

## 下一步

- 短期：只对 swing 做最小启动补偿，用于验证方向正确但幅度不足的问题。
- 数据：补录不同初始土体、dump 后土体状态、不同光照/遮挡下的启动和 return。
- 训练：提高 qpos/阶段信息权重，评估 `FPV + qpos`、qpos-only baseline、FPV 数据增强或阶段条件模型。
- 评估：每次 live 前固定 qpos 替换多组 FPV，先看动作是否稳定，再上真机。
