# Real One-Dig Policy Control Chain

本文档记录当前本地分支的目标：先在开发机把真实观测到 ACT 模型输出的链路打通，再同步到厂房真机。

## 当前策略

- 使用 `record_local_dev` 真机栈作为部署基础，新分支只增加 policy action source，不改变底层 CAN/control 语义。
- checkpoint bundle 放在本地 `policy_bundles/real_one_dig_v1/`，该目录被 `.gitignore` 忽略。开发机上可以用 symlink 指向 Repo A 的训练产物；真机 Jetson 上再用移动硬盘拷贝同名文件。
- bundle 至少包含 `policy_best.ckpt`、`dataset_stats.pkl`、`resolved_config.yaml`。`run_metadata.json` 可选但建议带上。
- 默认运行 `shadow_zero`：模型会读取真实 `FPV + qpos + qvel` 并预测 action，但返回给 backend 的动作是 `[0, 0, 0, 0]`。
- policy 测试默认 `recording.enabled: false`，不生成训练 HDF5，也不需要按 record-start。轻量 `steps.jsonl` 会记录 `qpos/qvel`、`policy_action`、`policy_assisted_action`、`safe_action`、`commanded_action`、guard/health/fault 和时间戳，方便动作没完成时复盘。
- 后续继续录制训练数据仍使用 `teleop_real_v1.yaml` 和按键 2；测试配置只在显式 `--record` 时才生成 HDF5。
- 只有 shadow 日志确认正常后，才把 `teleop.policy.output_mode` 改为 `control`，直接跑完整 one-dig 测试窗口；安全边界依赖 `safety.action_clip`、deadman、急停和人工接管，而不是 3 秒短窗。

## 死区助力

`teleop.policy.deadzone_assist` 用于处理“模型有动作意图但输出没有跨过液压死区”的情况。
默认 `enabled: false`，不会改变现有控制行为。启用后，动作处理顺序是：

```text
policy_action -> action_scale -> deadzone_assist -> guard -> control pump
```

也就是说，原始 `policy_action` 会原样保留；助力后的动作写到
`policy_assisted_action`，并作为 `control` 模式下返回给 guard 的动作。助力不会绕过
guard、deadman、急停或人工接管。

触发条件：

- 每个轴使用独立的正/负方向死区：`deadzone_positive` 和 `deadzone_negative`。
- 动作幅值达到 `trigger_fraction * deadzone` 才认为有动作意图。
- 同方向连续达到 `min_consecutive_steps` 才会助力，方向变化会重新计数。
- 真正助力时，动作被抬到 `deadzone + margin`，并受 `clip` 限制。

CLI live 行会显示当前状态：

- `assist=off`：配置未启用。
- `assist=idle`：配置已启用，但当前帧没有助力。
- `assist=swing+,boom-`：当前帧正在助力，且明确列出轴和方向。

`steps.jsonl` 和 HDF5 diagnostics 会记录：

- `policy_assisted_action`
- `policy_deadzone_assist_enabled`
- `policy_deadzone_assist_active`
- `policy_deadzone_assist_mask`
- `policy_deadzone_assist_axes`
- `policy_deadzone_assist_positive`
- `policy_deadzone_assist_negative`

首次启用前，应先用 `shadow_zero` 或离线动作 CSV 复盘确认：起步窗口是否改善，末段
`extra/wrong` 是否恶化。当前最需要警惕的是末段多动；不要把 assist 当作模型效果问题的
根治方案。

## 本地 Shadow Smoke

```bash
tb-receiver-real \
  --config testbed/testbed/configs/policy_real_one_dig_v1.yaml \
  --backend mock \
  --state-reader mock \
  --input policy \
  --num-episodes 1 \
  --max-steps 5 \
  --test-log-dir /tmp/real_one_dig_policy_shadow_smoke
```

这条命令只用 mock backend，不会触碰真机。成功后检查 `/tmp/real_one_dig_policy_shadow_smoke/*/steps.jsonl` 和 `summary.json`。

## 真机前置检查

- 确认真机侧 `policy_bundles/real_one_dig_v1/` 文件完整，并且 Python 依赖包含 `torch`、`torchvision`、`opencv-python`、`PyYAML`、`einops`、`IPython`。
- 先用 `output_mode: shadow_zero` 连接真实 `bridge_tcp`，只保存测试日志，不下发 policy action，也不保存训练 HDF5。
- 对比 `steps.jsonl` 里的 `policy_action`、`policy_assisted_action`、`commanded_action`、`qpos`、`qvel` 和现场 FPV，确认模型输出方向、幅度、助力状态和延迟合理。
- 进入 `control` 前，保证现场 deadman、急停和人工接管都可用；第一次完整测试仍不保存训练 HDF5，只保存 `steps.jsonl`。

## 不变的边界

- `raw_action`、`commanded_action` 和 policy diagnostics 只用于诊断，不修改原始录制数据。
- 模型输出仍是 `[swing, boom, stick, bucket]` 的 normalized action。
- 底层速度、阀控、CAN、安全状态仍由 `control/` 和 bridge 负责；policy source 只提供上层 4D command。
