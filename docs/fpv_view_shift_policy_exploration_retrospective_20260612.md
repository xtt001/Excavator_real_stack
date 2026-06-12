# FPV View Shift Policy 探索复盘（2026-06-12）

## 背景

本轮探索从真机现象开始：`fpv000` 在接近 home pose 或起步附近经常输出接近 0
的动作。这个现象不能只用整体 MAE 判断，因为真机液压系统存在死区；一个看起来
数值不小的动作如果没有跨过该轴该方向的死区，仍然可能不会让机器动起来。

本轮目标不是直接重录数据，而是在现有数据上完成三件事：

1. 找出已有离线测试为什么没有暴露真机问题。
2. 固定一套以后必须使用的模型效果分析方法。
3. 按这套方法比较当前候选模型、FPV 变换、view shift 拆分和 deadzone assist，
   判断是否已经有比原始 baseline 更好的版本。

相关原始分析文档：

- `docs/real_policy_live_test_fpv_failure_20260611.md`
- `docs/fpv_view_shift_after_episode26_20260611.md`
- `docs/policy_model_effect_eval_protocol.md`

主要结果目录：

- `artifacts/policy_effect_eval/deadzone_hdf5_20260612/`
- `artifacts/factor_isolation_20260612/`

## 最终结论

当前最好的已验证模型仍然是原始的 `FPV + qpos` baseline：

```text
runs/ckpts/real_excavation_act_20hz_v1
```

也就是没有调整 `vision_feature_scale/proprio_feature_scale`、训练和推理都使用原图
FPV 与 qpos 的版本。

其它尝试没有超过它：

- `fpv000_qpos20`：基本去掉 FPV，只靠 qpos，主动作段明显欠动作。
- `fpv025_qpos15`：主动作段可以，但起步/末段更容易多动，live-like 上有 early start。
- `downsample060`：all31 主动作有效率略高，但方向正确性更差、末段多动明显增加。
- `deadzone assist`：能把弱动作抬过死区，但同时会把错误时机的动作也抬过死区，
  不能作为模型效果问题的根因修复。

本轮探索的失败点是：基于“减少 FPV 依赖、增强 qpos 依赖、或用 deadzone assist
补动作”的方向，没有找到一个综合强于原始 baseline 的方案。它的价值是固定了以后
评估模型的正确方法，并确认当前问题更像“数据语义、视角漂移、qpos 漂移和末段标注
混在一起”的系统问题，而不是单个训练权重参数可以直接修好。

## 为什么原来的离线测试没有暴露问题

原有离线 replay 是 open-loop：每一步从 HDF5 读取记录下来的 qpos/qvel/FPV，
构造 observation，然后让 policy 输出动作，再和 HDF5 expert action 比较。
这个过程不会把 policy 动作回灌到机器状态里。

它适合回答：

- policy 平均动作误差多大。
- 每个 episode 的整体 MAE/RMSE/p95 如何。
- 曲线上 policy 和 expert 大致是否同形。

它不适合单独回答：

- 动作是否跨过液压死区。
- 起步能不能真正开始动。
- 末段能不能正确停下。
- 模型是不是输出了方向正确但幅度不足的动作。
- 模型是不是在 expert 不需要动作时继续输出有效动作。

旧的曲线图不是没用，而是没有“死区 + 局部窗口”的判定规则时，只能人工看形状，
不能稳定地给出模型好坏结论。所以本轮把模型效果分析口径改成：

1. 完整跑 episode，不用 `--max-steps` 当正式结论。
2. 从完整 `actions.csv` 中切出局部窗口。
3. 用每个关节正/负方向的死区判断有效动作。
4. 同时看主动作段、起步窗口和末段窗口。

## 固定对比方法

### 死区估计

死区来自现有 HDF5 的 `action -> qpos/qvel` 响应估计：

```text
artifacts/policy_effect_eval/deadzone_hdf5_20260612/deadzone_action_hdf5_estimate.json
```

方法是按轴、按正负方向统计动作发出后约 0.2s 内 qpos/qvel 是否出现同向响应。
数据不足的方向按用户要求默认使用 `0.5`。

| 轴 | 正方向死区 | 负方向死区 | 备注 |
| --- | ---: | ---: | --- |
| swing | 0.661 | 0.721 | HDF5 估计 |
| boom | 0.259 | 0.357 | HDF5 估计 |
| stick | 0.500 | 0.500 | 数据不足，默认值 |
| bucket | 0.408 | 0.508 | HDF5 估计 |

这不是最终硬件标定。后续如果有 `scripts/calibrate_axis_response.py` 的真机单轴标定，
应优先替换这张表。

### 窗口定义

固定分析三个窗口：

| 窗口 | 含义 | 主要用途 |
| --- | --- | --- |
| `start40` | episode 前 40 step | 判断是否无端起步或起步弱动作 |
| `longest_expert_effective_segment_gap5` | expert 跨死区的最长主动作段，允许 5 step 小间断 | 判断核心动作是否能执行 |
| `end80` | episode 最后 80 step | 判断是否末段多动、收不住或回收错误 |

关键指标：

- `policy_any_effective_pct`：policy 任一轴越过对应方向死区的帧比例。
- `same_axis_dir_effective_pct_of_expert_effective`：expert 有效时，policy 是否同轴同向有效。
- `extra_or_wrong_pct`：expert 不需要该轴该方向时，policy 是否仍输出了有效动作。

整体 MAE 仍保留，但只作为辅助指标，不再单独决定模型好坏。

## 候选模型与训练配置

| 模型 | 路径 | 关键差异 | best val loss |
| --- | --- | --- | ---: |
| `baseline_original` | `runs/ckpts/real_excavation_act_20hz_v1` | 原始 FPV + qpos，默认视觉/本体权重 | 0.1023 |
| `fpv000` | `runs/ckpts/real_excavation_act_20hz_v1_fpv000_qpos20` | `vision_feature_scale=0.0`, `proprio_feature_scale=2.0` | 0.1062 |
| `fpv025` | `runs/ckpts/real_excavation_act_20hz_v1_fpv025_qpos15` | `vision_feature_scale=0.25`, `proprio_feature_scale=1.5` | 0.1164 |
| `downsample060` | `runs/ckpts/real_excavation_act_20hz_v1_downsample060` | 训练/推理一致使用 `downsample_060` | 0.1140 |

这些模型使用的 31 个 train-ready episode split 是一致的：

```text
26, 29, 30, 32, 33, 34, 35, 36, 38,
39, 40, 41, 42, 43, 47, 48, 49, 50,
51, 53, 54, 55, 56, 57, 58, 59, 60,
61, 62, 63, 65
```

所以 baseline 不是靠更多 episode 取胜。

## 主要实验结果

### 1. All31 死区窗口对比

来源：

```text
artifacts/factor_isolation_20260612/downsample060_vs_existing_all31_deadzone_eval_20260612/deadzone_window_aggregate.csv
```

| 模型 | 主动作有效 | 主动作同向 | 主动作多/错 | 末段有效 | 末段同向 | 末段多/错 | 起步多/错 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_original` | 95.37 | 94.46 | 1.51 | 40.04 | 61.73 | 7.30 | 0.00 |
| `fpv000` | 84.96 | 82.56 | 2.89 | 25.56 | 36.29 | 7.90 | 0.00 |
| `fpv025` | 92.51 | 91.10 | 1.96 | 56.69 | 75.89 | 17.62 | 2.26 |
| `downsample060` | 95.74 | 93.39 | 2.92 | 56.94 | 75.37 | 19.19 | 0.00 |

解释：

- `fpv000` 明显欠动作。它不是最容易多动，但主动作执行弱。
- `fpv025` 和 `downsample060` 更积极，但末段多/错动作明显增加。
- `downsample060` 主动作有效率略高于 baseline，但方向正确性更低、多/错更多，
  末段多动也更严重，不能判定为更好。
- baseline 在主动作段不是最高的单一指标，但综合最稳。

### 2. Live-like 66..70 死区窗口对比

来源：

```text
artifacts/factor_isolation_20260612/downsample060_vs_existing_live_like_deadzone_eval_20260612/deadzone_window_aggregate.csv
```

| 模型 | 主动作有效 | 主动作同向 | 主动作多/错 | 末段有效 | 末段同向 | 末段多/错 | 起步多/错 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_original` | 97.04 | 96.35 | 0.69 | 53.00 | 82.30 | 18.25 | 0.00 |
| `fpv000` | 74.19 | 74.19 | 0.00 | 27.50 | 66.14 | 0.00 | 0.00 |
| `fpv025` | 95.49 | 94.68 | 0.80 | 60.75 | 92.73 | 23.25 | 20.00 |
| `downsample060` | 95.03 | 94.11 | 0.92 | 52.25 | 83.84 | 16.75 | 0.00 |

解释：

- live-like 上 baseline 主动作段最好，且没有起步窗口额外动作。
- `fpv000` 主动作段明显欠动作，符合真机上接近 0 的现象。
- `fpv025` 主段不错，但起步窗口出现 20% 额外有效动作，末段也更容易多动。
- `downsample060` 末段多动略低于 baseline，但主动作段下降；不能替代 baseline。

### 3. View domain 拆分

来源：

```text
artifacts/factor_isolation_20260612/view_domain_deadzone_model_comparison.csv
```

`episode_26..38` 是新视角早期，`39..65` 是新视角后期，`66..70` 更像 live-like。

| 数据域 | 模型 | 主动作有效 | 主动作同向 | 主动作多/错 | 末段多/错 |
| --- | --- | ---: | ---: | ---: | ---: |
| all31 early 26..38 | baseline | 95.51 | 94.58 | 1.33 | 6.39 |
| all31 early 26..38 | fpv025 | 92.58 | 91.50 | 1.48 | 20.00 |
| all31 early 26..38 | fpv000 | 83.42 | 81.12 | 2.53 | 14.17 |
| all31 late 39..65 | baseline | 95.32 | 94.41 | 1.59 | 7.67 |
| all31 late 39..65 | fpv025 | 92.48 | 90.94 | 2.16 | 16.65 |
| all31 late 39..65 | fpv000 | 85.59 | 83.15 | 3.04 | 5.34 |
| live-like 66..70 | baseline | 97.04 | 96.35 | 0.69 | 18.25 |
| live-like 66..70 | fpv025 | 95.49 | 94.68 | 0.80 | 23.25 |
| live-like 66..70 | fpv000 | 74.19 | 74.19 | 0.00 | 0.00 |

解释：

- view shift 真实存在，但 baseline 在 early/late 两个已训练子域都最稳。
- `fpv000` 在 live-like 显著掉下去，说明只靠 qpos 不足以覆盖外部视角/状态变化。
- `fpv025` 降低视觉权重没有让模型更稳，反而保留了末段和起步风险。

### 4. Early/Late domain split 训练

这组实验来自原始 FPV 文档计划：按视角子域做 held-out，而不是随机 val。

| 模型 | train split | eval split | best val loss | replay MAE |
| --- | --- | --- | ---: | ---: |
| `domain_early_train_late` | 26..38 | 39..65 | 0.3057 | 0.0846 |
| `domain_late_train_early` | 39..65 | 26..38 | 0.1346 | 0.0825 |

解释：

- 早期视角训练到后期视角明显困难，支持 view shift 担忧。
- 后期视角训练到早期视角相对好一些。
- 这说明随机 split 会高估泛化能力，但 domain split 本身没有产出比 baseline 更好的部署模型。

### 5. 固定 qpos 替换 FPV

来源：

```text
artifacts/factor_isolation_20260612/fpv_nearest_qpos_sensitivity_existing_grid_model_aggregate.csv
artifacts/factor_isolation_20260612/fpv_swap_nearest_qpos_model_aggregate.csv
```

固定目标 qpos/action，只替换不同 episode 的 FPV。

| 模型 | 条件 | cases | mean action diff | p95 step max diff | max step max diff |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | all swaps | 22 | 0.082 | 0.617 | 0.829 |
| fpv025 | all swaps | 22 | 0.068 | 0.555 | 0.756 |
| fpv000 | all swaps | 22 | 0.000 | 0.000 | 0.000 |

更严格的 nearest-qpos counterfactual：

| 模型 | image 条件 | cases | mean diff | p95 step max diff | start40 policy eff | end80 policy eff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | swapped image | 4 | 0.092 | 0.738 | 10.63 | 63.13 |
| baseline | self image | 2 | 0.000 | 0.000 | 0.00 | 50.63 |
| fpv025 | swapped image | 4 | 0.071 | 0.597 | 75.00 | 80.00 |
| fpv025 | self image | 2 | 0.000 | 0.000 | 48.75 | 51.25 |

解释：

- baseline 和 fpv025 都会明显受 FPV 变化影响，说明视觉确实在驱动动作。
- `fpv000` 完全不受 FPV 影响，这符合配置，但也解释了它为什么无法利用画面修正 qpos 漂移或阶段不确定性。
- FPV 不是唯一问题，但它是动作变化的重要来源。

### 6. qpos 微扰

来源：

```text
artifacts/factor_isolation_20260612/qpos_perturb_live_like66_70_aggregate.csv
```

结论：

- baseline 对小幅 qpos 微扰有反应，但影响通常小于固定 qpos 换 FPV。
- `fpv025` 对 `stick` 或全轴 `0.05 rad` 级别扰动更敏感，最大 step diff 可到
  `0.31..0.42`。
- 这说明 qpos/IMU 漂移会影响模型，但当前最大的问题不是“只要提高 qpos 权重就能解决”。

### 7. FPV transform

来源：

```text
artifacts/factor_isolation_20260612/non_collection_experiment_summary_20260612.md
```

Baseline original 在 live-like 上做推理时图像变换：

| transform | 主动作有效 | 主动作同向 | 主动作多/错 | 末段多/错 | 起步多/错 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 97.04 | 96.35 | 0.69 | 18.25 | 0.00 |
| downsample_060 infer-only | 97.12 | 96.55 | 0.57 | 11.00 | 0.00 |
| center_zoom_085 | 84.99 | 80.23 | 4.75 | 2.75 | 60.00 |
| center_zoom_085_blur | 88.78 | 85.92 | 2.86 | 0.00 | 60.00 |
| center_zoom_075 | 58.41 | 20.84 | 37.57 | 0.00 | 20.00 |

解释：

- 推理时 `downsample_060` 对 baseline 有离线吸引力：主动作不掉，末段多动下降。
- 但这不是训练/推理一致的方案，不能直接作为默认部署。
- center zoom 系列虽然降低末段多动，但严重破坏主动作段或引入起步多动，不可用。

训练/推理一致的 `downsample060` 模型没有超过 baseline：

| eval | replay MAE | 结论 |
| --- | ---: | --- |
| all31 | 0.0610 | 主动作有效略高，但方向更差、末段多动更重 |
| live-like 66..70 | 0.0694 | 末段多动略低于 baseline，但主动作段略差 |

### 8. Deadzone assist 全局回测

来源：

```text
artifacts/policy_effect_eval/deadzone_hdf5_20260612/assist_global_backtest_model_comparison.csv
artifacts/policy_effect_eval/deadzone_hdf5_20260612/assist_threshold_sweep_tradeoff.csv
```

默认 assist 思路：当模型输出达到死区一半，且同方向连续稳定后，把动作抬到
`deadzone + margin`。

默认 `trigger_fraction=0.5, min_consecutive_steps=2` 的结果：

| dataset | model | window | raw any | assisted any | raw extra | assisted extra | assist active |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| all31 | baseline | main | 95.37 | 99.50 | 1.51 | 1.78 | 16.10 |
| all31 | baseline | start40 | 0.00 | 29.27 | 0.00 | 28.39 | 29.27 |
| all31 | baseline | end80 | 40.04 | 73.19 | 7.30 | 28.19 | 35.12 |
| live-like | baseline | main | 97.04 | 100.00 | 0.69 | 1.26 | 24.03 |
| live-like | baseline | start40 | 0.00 | 39.00 | 0.00 | 39.00 | 39.00 |
| live-like | baseline | end80 | 53.00 | 81.00 | 18.25 | 45.50 | 28.00 |

解释：

- assist 能提高主动作段有效率，但 baseline 主动作段本来已经很高。
- assist 会显著放大起步和末段的额外有效动作。
- 阈值 sweep 显示，提高 `trigger_fraction` 或 `min_consecutive_steps` 可以减少副作用，
  但也会让 assist 的收益变得很小。例如 live-like baseline 在 `0.95/20` 时，主动作
  提升约 `0.46`，末段 extra 仍增加约 `2.25`。
- 因此 assist 只能作为带明确日志的诊断/部署保护候选，不能替代模型和数据问题修复。

## 为什么这些探索没有产出更好模型

### 1. “只靠 qpos”方向失败

`fpv000` 的思路是去掉视觉干扰、提高 qpos 权重。但我们的 qpos 不是绝对可靠的世界状态：

- IMU 存在误差积累。
- 同一个物理位置不严格对应同一个 qpos 组合。
- qpos 有测量波动。
- 数据中同一 qpos 附近可能属于不同阶段。

结果是：`fpv000` 不会被 FPV 干扰，但也不能靠画面修正阶段和位置，live-like 主动作
有效率掉到 `74.19%`。

### 2. “降低 FPV 权重”方向没有解决起止判断

`fpv025` 保留少量视觉，理论上希望降低 FPV 过敏。实际结果是：

- 主动作段比 `fpv000` 好，但仍低于 baseline。
- 起步窗口在 live-like 有 `20%` 额外有效动作。
- 末段多/错动作 all31 为 `17.62%`，live-like 为 `23.25%`。

这说明问题不是“视觉太强，所以压低视觉即可”。视觉确实有噪声，但模型仍需要视觉来
判断场景；简单调权重会把阶段判断变得更不稳定。

### 3. “图像低通/降采样”没有稳定超过 baseline

训练/推理一致的 `downsample060` 没有赢。它更容易跨过死区，但不是更准确：

- all31 主动作有效率 `95.74` 略高于 baseline `95.37`。
- all31 主动作同向降到 `93.39`，多/错升到 `2.92`。
- all31 末段多/错升到 `19.19`，比 baseline 的 `7.30` 高很多。

这说明图像低通可能降低某些视觉细节噪声，但也改变了模型对动作阶段的判断。

### 4. Assist 放大了模型的错误意图

assist 的前提是“模型输出达到死区一半代表有正确动作意图”。回测证明这个前提不够稳：

- 主动作段里它通常有帮助。
- 但起步和末段里，它会把本来还没越过死区的错误动作也抬成真机有效动作。

所以 assist 不能解决“什么时候该动、什么时候该停”的问题，只能解决“方向已经对、时机也对，
但幅度略低”的小问题。

### 5. 数据末段语义本身不干净

当前录制流程是：操作者回到出发点附近后按 gohome，并没有明确暂停输入或录一段稳定
idle。这样末段数据里混有：

- 人工回到出发点附近的最后动作。
- gohome 或接近 home 的回收动作。
- 没有明确“任务完成后应该停住”的静止标签。

模型看到的末段不总是“停”，而可能是“继续回收/修正”。这解释了为什么多个模型都会有
末段多动，只是程度不同。

### 6. View shift 是问题之一，但不是唯一问题

原始 FPV 文档指出 `26..38`、`39..65`、`66..70/live` 有明显视角子域差异。
本轮实验支持这个判断：early->late domain split 明显困难，固定 qpos 换 FPV 会让
动作显著变化。

但 baseline 在 all31 early/late 两个子域内仍然最稳，说明当前失败不是只靠“换一个 FPV
预处理”就能解决。它是 FPV 视角变化、qpos 漂移、数据阶段语义和末段采集流程共同造成的。

## 对后续工作的约束

1. 模型效果必须按 `docs/policy_model_effect_eval_protocol.md` 的 deadzone/window
   方法报告；不能只用 MAE 或曲线图做最终结论。
2. 每次候选模型至少报告 all31 和 live-like 66..70 两套结果。
3. 必须报告 `start40`、主动作段和 `end80`，尤其是 `extra_or_wrong`。
4. 使用 FPV 的模型必须做固定 qpos 替换 FPV 检查。
5. assist 只能在日志里明确标出状态和轴向，不能把 assisted action 当作原始模型能力。
6. 在没有新数据前，不建议把默认模型从 `baseline_original` 切到 `fpv000`、`fpv025`
   或训练一致的 `downsample060`。

## 下一步建议

短期不重新采集时：

1. 保持 `baseline_original` 作为默认最佳候选。
2. 对 baseline 的推理时 `downsample_060` 做更严格离线 replay，可作为谨慎候选，
   但不能直接替代默认模型。
3. 针对末段多动尝试“基于阶段证据的末段抑制”，不能用盲目的时间阈值。
4. 继续用 fixed-qpos FPV swap 作为 live 前检查。

未来重新采集或新 IMU 稳定后：

1. 固定相机安装，记录每次相机姿态变化。
2. 每条任务结束后增加明确 idle/stop 片段，再 gohome 或结束记录。
3. 单独采集启动、停止、return 和 home 附近动作，补齐当前最弱窗口。
4. 用真机单轴标定覆盖 HDF5 估计死区。
5. 记录阶段或任务状态标签，减少相近 qpos 下不同动作阶段的歧义。

## 本轮产出

代码和配置层面已经沉淀：

- `docs/policy_model_effect_eval_protocol.md`
- `scripts/deadzone_window_eval.py`
- `testbed/testbed/policies/deadzone_eval.py`
- `testbed/testbed/data/image_transforms.py`
- `testbed/testbed/configs/act_real_20hz_v1_downsample060.yaml`
- `testbed/testbed/configs/act_real_20hz_v1_domain_early_train_late_val.yaml`
- `testbed/testbed/configs/act_real_20hz_v1_domain_late_train_early_val.yaml`
- `data/fpv_view_shift_experiments_20260611/*`

本复盘文档的作用是把失败路径和结论固定下来：本轮没有找到更好的部署模型，但已经
把“为什么原测试没暴露问题、以后如何比较模型、为什么几个直觉修法失败”记录成可复用方法。
