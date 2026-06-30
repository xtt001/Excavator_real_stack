# GMSL HFOV110 ACT 纹理泛化风险记录

日期：2026-06-30

本文记录当前 GMSL `virtual_rectilinear` 处理方案下，ACT 训练可能遇到的沙面纹理、光照和单场景过拟合风险，以及现场采集和离线验证时应保留的处理思路。本文不替代相机标定和实时预处理文档，只补充策略学习层面的风险预期。

相关前提：

- 图像投影：`virtual_rectilinear`
- HFOV：`110 deg`，作为当前训练和推理的固定处理决议
- 输出尺寸：以预处理 manifest 为准，当前目标为 `384x216`
- 训练和推理必须使用同一套 transform version，包括相机顺序、pitch/yaw、resize、颜色空间和归一化
- 原始 fisheye/raw oriented 图像继续保留为复查和重新生成派生输入的证据

参考文件：

- `docs/gmsl_four_camera_calibration_guide_20260630.md`
- `docs/gmsl_multi_camera_act_input_plan_20260629.md`
- `docs/real_policy_runtime_and_generalization_issues.md`
- `docs/policy_model_effect_eval_protocol.md`
- `artifacts/gmsl_virtual_rectilinear_trial_20260630/virtual_rectilinear_metadata.json`

## 1. 当前风险判断

当前 HFOV110 处理后，画面清晰度和任务几何信息总体足够。主要风险不是“看不清”，而是 ACT 在小数据条件下过度利用稳定但脆弱的视觉线索：

- 固定沙纹、铲痕、小沙丘阴影被模型当成动作阶段信号。
- 单一沙堆形状或固定入土区域导致模型记住局部纹理，而不是学习铲斗和土堆的几何关系。
- 固定室内墙面、光照、线束或车体边缘成为 shortcut。
- 某一路辅助相机如果长期被机械结构或无关背景主导，可能增加过拟合空间。

因此，“人为填沙引入纹理差异”是有价值的，但它应该作为采集 protocol 的一部分，而不是后处理补救。目标是让模型看到同一动作阶段对应多种沙面纹理、堆形和光照状态，降低纹理和动作阶段之间的偶然绑定。

## 2. 现场沙面扰动原则

推荐做物理扰动，优先级高于单纯图像增强。

每条或每小批 episode 前，可以人为改变：

- 沙堆整体形状：高度、坡度、左右偏移、前后位置。
- 沙面纹理：刮平、留下不同方向铲痕、局部松散、局部压实。
- 入土区域：不要每次都从完全相同的纹理和同一条沟槽进入。
- 装载残留：保留或清理上一轮留下的局部小堆，但不要形成固定流程。
- 起始接触状态：覆盖空斗接近、轻触、局部入土前的多种画面。

注意事项：

- 不要把沙面每次整理成同一种“标准纹理”，否则会制造新的固定 cue。
- 不要让某种纹理只出现在成功轨迹、另一种纹理只出现在失败或异常轨迹中。
- 不要只改变表面花纹而不改变任务几何。ACT 真正需要泛化的是铲斗、土堆和动作时序关系。
- 如果现场必须重复同一工况，应记录这是受控重复，不把它当作泛化数据。

建议最小做法：

- 每 5-10 条 episode 改一次沙堆形状。
- 每条 episode 前避免完全复刻同一初始沙纹。
- 每个采集批次至少包含刮平、已有铲痕、局部小堆、局部压实四类沙面状态。
- 每次明显改变沙面后保存起始 contact sheet 或至少保存四路起始帧。

## 3. 光照和曝光扰动

光照风险和纹理风险同等重要。相同沙面在阴影、反光、曝光漂移下会形成不同视觉分布，ACT 可能把亮暗区域当成动作阶段信号。

建议：

- 采集时记录曝光/增益是否固定。
- 如果现场可控，优先固定曝光和白平衡，减少推理时漂移。
- 如果现场光照不可控，应主动覆盖几类光照状态，例如强光、弱光、局部阴影和反光。
- 每个光照状态下都要覆盖多个动作阶段，而不是只在某个阶段出现。

不建议：

- 只靠后期亮度增强弥补现场曝光漂移。
- 训练时使用强烈颜色增强，但推理时画面风格完全不同。
- 把局部阴影稳定保留在同一动作阶段。

## 4. 图像增强边界

图像增强可以辅助泛化，但不能替代现场物理扰动。

可优先尝试：

- 小幅 brightness / contrast / gamma 扰动。
- 小幅 saturation / hue 扰动。
- 轻微 JPEG quality 扰动。
- 轻微 noise 和 blur。
- 局部 soft shadow augmentation，用于弱化固定小阴影的影响。

需要谨慎：

- 强 random crop、translation 或 scale。HFOV110 虚拟视角已经是训练和推理契约，过强几何扰动会破坏相机几何语义。
- 大角度旋转或透视变换。它们会制造真实相机不会看到的图像。
- 水平翻转。左右动作语义可能被破坏，除非同步变换 action label。
- MixUp / CutMix。行为克隆图像混合后的 action label 未必有物理意义。

推荐第一阶段组合：

```text
HFOV110 deterministic transform
+ fixed camera order
+ physical sand/pile variation
+ light color augmentation
+ mild blur/noise/JPEG augmentation
```

## 5. 采集元数据建议

为了后续能判断模型是否依赖纹理，需要把现场扰动记录成可检索信息。最低限度可以先用人工文本或 CSV 记录，不必一开始改 HDF5 schema。

建议字段：

| 字段 | 含义 |
| --- | --- |
| `transform_version` | HFOV110 处理版本、相机顺序、pitch/yaw、resize |
| `sand_surface_variant` | `flat`、`raked`、`tracked`、`loose`、`compacted` 等 |
| `pile_shape_variant` | `low`、`high`、`left_bias`、`right_bias`、`front_bias` 等 |
| `lighting_variant` | `normal`、`shadow`、`bright`、`reflective`、`low_light` 等 |
| `camera_visibility_note` | 遮挡、泥土、反光、模糊、线束干扰 |
| `start_frame_path` | episode 起始帧或 contact sheet |

这些字段的目的不是追求标注精细，而是让离线分析可以按纹理/堆形/光照分组，而不是只能随机 train/val split。

## 6. 离线验证口径

不能只看 val loss 判断泛化。至少保留以下 gate：

1. 固定 qpos / 多 FPV 替换 replay
   - 保持 qpos/action 轨迹不变，替换不同沙面、光照或 camera variant 的图像。
   - 检查模型是否因为画面纹理变化而提前、滞后或输出保守均值动作。

2. 按扰动分组的 held-out split
   - 不只随机切分。
   - 至少做按 `sand_surface_variant`、`pile_shape_variant` 或 `lighting_variant` 留出的验证。

3. 任务窗口分段统计
   - 分开看起步、入土、装载、回收/末段。
   - 不允许只报告整体 MAE 或整体 action loss。

4. 视角消融
   - 先保留主视角 baseline。
   - 再比较主视角 + 辅助视角。
   - 如果辅助视角只带来训练 loss 下降，但 held-out 纹理/光照变差，不应默认加入。

失败信号：

- 换一组沙面纹理后，qpos 近似但动作幅度明显变小。
- 末段或 return-like qpos 下，画面纹理诱发多余动作。
- 模型对固定墙面、线束、车体边缘或局部阴影的敏感度高于铲斗/土堆几何。
- 随机 val split 表现好，但按沙面/光照留出的 split 明显变差。

## 7. 当前建议结论

短期建议采用以下策略：

- HFOV110 作为固定图像处理口径，不再在 90/110/130 之间切换。
- 主视角先做单路 baseline，再逐步加入辅助视角做消融。
- 现场采集时主动引入沙面、堆形和光照变化。
- 图像增强只做轻量辅助，不改变 HFOV110 几何契约。
- 后续模型验收必须包含固定 qpos / 多 FPV gate 和按纹理/光照分组的 held-out 检查。

核心判断：人为填沙引入纹理差异能降低过拟合风险，但必须和任务几何、光照覆盖、元数据记录以及离线消融一起使用。否则它只能增加样本数量，不能保证 ACT 学到真正可迁移的铲斗-土堆关系。
