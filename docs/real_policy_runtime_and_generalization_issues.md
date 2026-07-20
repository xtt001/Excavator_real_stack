# 真机 policy 运行速度与泛化问题总结

本文汇总近期真机 policy shadow/control 测试中暴露出的主要问题、已经得到的线索、可能的解决思路和建议探索顺序。讨论对象是当前 real stack 中的 ACT policy、FPV 输入、qpos/qvel 输入、gohome 起始姿态以及 Jetson 部署链路。

## 1. 当前主要问题

### 1.1 Jetson 上 policy 循环达不到 50Hz

现象：

- 目标控制频率是 50Hz。
- Jetson 上实际 policy 运行速度不到 20Hz。
- 相机原始帧率约 30Hz，当前链路中存在插帧到 50Hz 的情况。

当前代码结构中的关键点：

- `testbed/testbed/actions/policy.py` 的 `PolicyActionSource.next_action()` 每个 step 都会调用 `policy.predict()`。
- `testbed/testbed/policies/act/adapter.py` 的 `ACTAdapter.predict()` 每次都会重新构造图像 tensor、做归一化并执行完整模型前向。
- `testbed/testbed/configs/policy_real_one_dig_v1.yaml` 中 `real.control_pump.enabled` 当前默认是 `false`。
- 因此 policy 推理、观测读取、控制发送、日志记录很可能串在同一个 50Hz 主循环里。只要模型前向或图像处理超过 20ms，整个循环就会掉频。

初步判断：

- 这个问题不应该优先理解成“必须让视觉模型本身 50Hz 前向”。
- 更合理的目标是“控制指令保持 50Hz 下发，policy action 以可承受频率更新”。
- 对 30Hz 相机输入而言，每 20ms 强制完整视觉推理还会重复处理相同或插帧后的画面，收益有限。

### 1.2 FPV 对局部画面细节非常敏感

已做消融实验：

- 训练 FPV + 训练 qpos/qvel：能产生接近训练数据原始分布的 action。
- 训练 FPV + 实时 qpos/qvel：也能产生接近训练数据原始分布的 action。
- 实时 FPV + 训练 qpos/qvel：不能产生合理 action。
- 实时 FPV + 实时 qpos/qvel：不能产生合理 action。

结论：

- qpos/qvel 和 gohome 误差不是当前 action 崩掉的一号嫌疑。
- 实时 FPV 一旦进入模型，action 分布就偏离训练分布。
- 已排除 FPV 预处理和传输格式问题，包括 RGB/BGR、shape、归一化、传输链路等。

进一步现场观察：

- 训练 FPV 与实时 FPV 的最大差异不是格式，而是土体细节。
- 训练画面中存在几个约 5cm 的小沙丘，实时画面中没有。
- 这些小沙丘造成明显局部阴影差异。

初步判断：

- 模型很可能把小沙丘阴影、局部纹理或局部暗区当作可靠视觉特征。
- 由于训练数据只有 9 条，这类 spurious cue 很容易被模型记住。
- 一旦实时画面缺少这些阴影，输入就变成 out-of-distribution，导致 action 分布崩掉。

### 1.3 gohome 只能回到大致位置

已知条件：

- 每个轴按百分比计算的回位误差不同，但都在 0.05 以内。
- gohome 后直接开始 episode。
- qpos 视觉上误差不大，单轴最大差一般不到 0.1。
- 一个“轮次”指一个 episode。
- 目标场景是同一场景、大致形状接近的土堆、基本一致的 gohome 起始姿态。

判断：

- gohome qpos 误差会影响泛化，但在当前条件下不是首要问题。
- 如果训练数据覆盖了这些自然回位偏差，它反而可以作为有益的初始状态扰动。
- 真正风险在于训练数据只有 9 条，模型可能记住少数几个起点附近的 FPV/qpos 组合。
- 由于 qpos/qvel 会根据 dataset stats 标准化，样本少时某些轴的 std 可能较小，真实小偏差在归一化空间中可能被放大。

## 2. 运行速度问题的解决思路

### 2.1 启用 control pump，解耦控制频率和推理频率

优先级最高。

思路：

- 使用 `RealActionPump` 在后台以 50Hz 重复发送最新 safe action。
- policy 主循环只负责读取观测、计算新 action、更新给 pump。
- 如果 policy 当前只跑到 20-30Hz，底层控制仍然保持 50Hz heartbeat。

优点：

- 不改变模型结构。
- 不降低推理精度。
- 与当前框架已有设计一致。
- 能避免模型推理慢导致控制桥 watchdog 或液压控制链路断供。

注意：

- 需要区分“控制发送 50Hz”和“模型前向 50Hz”。
- 如果业务要求必须每 20ms 都有全新模型输出，则仅启用 pump 不够。但对当前 30Hz FPV 输入而言，这个要求本身需要重新评估。

### 2.2 异步 policy worker

思路：

- 单独线程或进程持有最新 observation 并运行模型。
- 控制链路只读取最新 policy action。
- 如果新 action 未完成，则继续使用上一帧 action。
- 如果 action age 超过阈值，则 guard 归零。

优点：

- 避免推理排队造成延迟堆积。
- 能把 policy 更新率、观测采样率、控制发送率分开管理。
- 便于记录 `policy_action_age_ms`、`policy_inference_latency_ms`、`policy_update_hz` 等诊断指标。

### 2.3 按图像时间戳去重

思路：

- 对同一个 `image_timestamp_ns` 的 FPV，不重复执行完整视觉前向。
- 如果实时 qpos/qvel 更新但图像未更新，可以考虑缓存视觉 backbone feature，只更新低维状态相关部分。

适用原因：

- 相机真实约 30Hz，控制目标 50Hz。
- 插帧或重复帧不应该触发完整视觉模型计算。

风险：

- 如果模型结构没有拆分视觉 backbone 和后续 transformer，直接做 feature cache 需要改模型前向接口。
- 先做“同图像时间戳跳过 policy 更新，沿用上一 action”更简单。

### 2.4 无损或近无损推理优化

可探索项：

- 将 `torch.no_grad()` 改为 `torch.inference_mode()`。
- 将 `model.eval()` 放到加载后，只做一次。
- 启动时做 warmup。
- 固定输入尺寸时开启 `torch.backends.cudnn.benchmark = True`。
- 预分配图像和 proprio tensor，减少每步 numpy 到 torch 的重复分配。
- 使用 pinned memory 或更直接的 tensor 构造路径。

需要确认的边界：

- FP16/TensorRT 会带来数值差异。若“推理精度不降低”要求严格 bit-level 或 FP32 输出一致，则不应作为第一阶段方案。
- 如果接受数值小差异但动作行为一致，可以单独评估 TensorRT/FP16。

#### 四相机共享 backbone 合批

ACT 的视觉 backbone 在不同相机之间共享权重。推理阶段把
`(batch, camera, channel, height, width)` 折叠为
`(batch * camera, channel, height, width)`，可以用一次 backbone 调用处理所有
相机，再恢复相机维度并按原 camera order 拼接特征。该路径：

- 只在没有 expert action 输入的推理前向中启用；训练和 validation 继续使用原逐相机路径；
- 不改变 checkpoint、模型参数、相机顺序、Transformer token 布局或 temporal aggregation；
- 不需要重新训练，但不同 batch shape 可能让 CUDA 选择不同卷积 kernel，因此不承诺 bitwise FP32 一致；
- 上机前仍需比较 raw action chunk、deadzone direction class、temporal aggregation 后动作以及持续负载延迟。

开发机 RTX 5070 Ti 的合成输入 smoke 使用 `batch=1`、四相机、
`216x384`、ResNet18、hidden dim 512：完整随机初始化 ACT 前向 median 从
`8.37 ms` 降至 `5.24 ms`，约 `1.60x`；归一化 action chunk 的
`max_abs_diff` 为 `2.23e-4`。已有双相机 E52 checkpoint smoke 的 median 从
`5.27 ms` 降至 `4.19 ms`，约 `1.26x`，归一化 action chunk
`max_abs_diff` 为 `4.35e-5`。这些数字只用于证明本地实现有实际收益和小量数值差异，
不能外推 Jetson AGX Orin 的加速比例，也不是现场动作等价结论。

#### 可切换 FP16 推理与精度检查

Runtime policy 配置支持 `inference_precision: fp32|fp16`，默认和当前现场配置均保持
`fp32`。`fp16` 只包围 ACT 神经网络前向；temporal aggregation、intent sigmoid、
动作反归一化和 CPU 输出继续使用 FP32。它不改变 checkpoint，也不需要重新训练。

在启用 `fp16` 现场控制前，使用同一 checkpoint 和同一段已录制观测运行：

```bash
PYTHONPATH=.:testbed python scripts/compare_act_inference_precision.py \
  --bundle-dir /path/to/policy_bundle \
  --episode-file /path/to/episode.hdf5 \
  --deadzone-json /path/to/deadzone.json \
  --device cuda \
  --max-steps 200 \
  --warmup-steps 20 \
  --require-deadzone-equivalence \
  --output-json artifacts/act_precision/fp32_vs_fp16.json
```

报告同时比较 temporal aggregation 后的执行动作、每次前向反归一化后的 raw action
chunk、deadzone direction class 和 P50/P95 latency。该检查是 recorded-observation
teacher-forced replay，只能验证数值和动作语义是否保持，不能替代真机闭环验证。动作
误差阈值不在脚本中猜测；需要门控时，依据当前控制容差显式传入
`--max-action-abs-diff`。性能验收默认要求 P95 latency 至少不退化，可用
`--min-p95-speedup` 提高目标，但不应为了让报告通过而降低到 `1.0` 以下。

开发机 RTX 5070 Ti 上，已有双相机 E52 checkpoint 对 episode 91 的 100 个连续步骤
实测结果是：FP32 P50/P95 为 `4.77/4.88 ms`，FP16 为 `4.84/5.12 ms`，FP16
反而更慢；聚合动作最大绝对差为 `1.09e-4`，400 个执行动作轴和 8,000 个 raw
chunk 动作轴均没有发生 deadzone class 改变。因此该 checkpoint 在开发机上通过了
本次语义一致性检查，但没有通过性能门槛，不能据此默认启用 FP16。Orin 和四相机
checkpoint 必须分别实测。

当前成功率最高的四相机 camera-role checkpoint 在同一开发机上的 200 步测试中，
逐相机 FP32 的 P95 为 `10.27 ms`，共享 backbone 合批后的 FP32 P95 为
`6.67 ms`，约 `1.54x`；合批 FP16 P95 为 `5.73 ms`。合批 FP32 相对逐相机
FP32 在 800 个聚合执行动作轴和 16,000 个 raw chunk 轴上均没有 deadzone class
变化。FP16 虽然没有改变 800 个聚合执行轴，却改变了 16,000 个 raw chunk 轴中的
3 个，因此严格 raw-chunk 等价门槛不通过，现场配置继续保持 FP32。

进一步对 20 条 validation episode 分布采样 500 个观测：TF32 加
`torch.compile(mode="reduce-overhead")` 将本地稳态 P95 降至约 `3.93 ms`，但
40,000 个 raw chunk 轴中有 5 个 deadzone class 变化；FP16 有 8 个。两者在这组
样本的 2,000 个 temporal-aggregation 执行动作轴中均没有类别变化。这只能说明聚合
在当前离线样本中吸收了临界数值差异，不能证明 Orin 或真机闭环等价。当前相机合批
是默认候选；FP16、TF32 和 compile 仍应作为显式实验开关。

分段 profile 显示，逐相机推理时 backbone 约占 wall time 的 `52.7%`；合批后
Transformer 上升为最大阶段，约占 `53.6%`。因此后续模型侧优化应优先针对
Transformer 的线性层和 attention 矩阵运算，而不是继续压缩已经降到次要位置的
相机预处理。

### 2.5 减少日志和 I/O 阻塞

现象风险：

- 轻量 `steps.jsonl` 每步写入并 flush，如果写外置盘，50Hz 时可能阻塞主循环。

建议：

- 测速时先关闭高频日志或改为批量 flush。
- 记录每步耗时拆分：`read_state_ms`、`policy_ms`、`guard_ms`、`send_ms`、`log_ms`。
- 不要只看整体 Hz，否则难以判断瓶颈在模型、图像、TCP、控制发送还是 I/O。

## 3. FPV 泛化问题的解决思路

### 3.1 训练和推理使用同一套确定性图像预处理

原则：

- 不能只在实时推理时修改画面。
- 训练和测试必须走同一套预处理，否则会产生新的输入分布差异。

目标：

- 不是把实时画面“修成训练画面”。
- 而是把训练和实时画面都压缩成更粗、更不依赖局部阴影和土面微纹理的视觉表示。

可选预处理：

1. 固定 ROI crop 或 mask
   - 如果小沙丘阴影位于任务无关区域，可以裁掉或遮住。
   - 训练和推理必须完全一致。
   - 风险最低、可解释性最好。

2. 轻微 blur 或 downsample-upsample
   - 用小核 Gaussian blur，或先降采样再升采样。
   - 目标是弱化 5cm 小起伏、局部纹理和阴影边缘。
   - 不能过强，否则会损失铲斗边缘、土堆大形状和关键几何关系。

3. 光照归一化
   - 在亮度通道上估计大尺度 illumination map。
   - 对图像做除法或减法校正，减少局部阴影影响。
   - 比简单 brightness/contrast 更针对阴影差异。

4. 颜色弱化
   - 降低 saturation，或只使用亮度通道。
   - 需要谨慎，若颜色对铲斗、土体、背景分离有帮助，过度弱化会损失信息。

推荐第一阶段组合：

- 固定 ROI/mask。
- 轻微 blur 或 downsample-upsample。
- 配合训练时局部 shadow augmentation。

### 3.2 训练时加入图像增强

原则：

- 训练时随机增强，推理时不随机增强。
- 增强应覆盖训练和实时之间的差异，而不是制造不真实的几何变化。

推荐增强：

- brightness、contrast、gamma。
- saturation、hue 小幅扰动。
- JPEG quality 扰动。
- 轻微 noise。
- 轻微 blur。
- 小幅 random crop、translation、scale。
- 局部 shadow augmentation：
  - 在土体区域随机生成软边缘暗斑或亮斑。
  - 使用椭圆或多边形 mask。
  - mask 边缘做 Gaussian blur。
  - 随机强度，例如暗 10%-40%。

不建议优先使用：

- 随机水平翻转：会改变 swing 左右语义，除非同步变换 action label。
- 大角度旋转或强透视变换：会破坏 FPV 几何关系。
- MixUp/CutMix：行为克隆中图像混合后的 action label 不一定物理合理。
- 只在推理时做强滤波或 CLAHE：会改变模型输入分布。

### 3.3 冻结或半冻结视觉 backbone

原因：

- 只有 9 条 episode 时，视觉 backbone 很容易记住局部阴影、纹理和小沙丘形状。

可探索方案：

- 冻结 ResNet backbone，只训练 transformer 和 action head。
- 或者 backbone 使用极低学习率，后续层使用正常学习率。
- 对比冻结与不冻结的 shadow/control 结果。

预期：

- 降低模型对局部纹理的记忆能力。
- 迫使模型更多利用较稳定的高层几何关系和 qpos/qvel。

### 3.4 补录反事实数据

比随机补录更高效。

建议录制覆盖：

- 有小沙丘阴影的 episode。
- 没有小沙丘阴影的 episode。
- 小沙丘位置轻微变化的 episode。
- 土面被抹平但动作目标一致的 episode。
- 光照略有差异的 episode。

目标：

- 让同一动作目标对应多种局部阴影和土面微形态。
- 让模型无法把某个固定阴影当成唯一动作锚点。

## 4. gohome 误差与数据量建议

在当前目标条件下：

- 同一场景。
- 土堆大致形状接近。
- gohome 起始姿态基本一致。
- 单轴 qpos 最大差一般不到 0.1。
- 每轴百分比误差小于 0.05。

判断：

- gohome 误差不是当前 action 崩掉的主因。
- 但训练数据太少时，gohome 偏差仍可能被归一化后放大。
- 建议保留真实 gohome 偏差，不要每次强行人工修到完全一致，否则数据分布更窄。

如果不做任何预处理和增强，只靠录制数量提升：

- 20-30 条有效 episode：通常能明显比 9 条稳定，适合第一轮验证。
- 40-60 条有效 episode：同一场景、相似土堆、相近 gohome 起点下，可能开始具备可用泛化。
- 80-100 条有效 episode：更适合覆盖光照、小土面变化、自然 gohome 偏差。

更有效的录制策略：

- 不要只录尽量一模一样的轨迹。
- 应主动覆盖真实 gohome 后的不同 qpos 起点。
- 在安全允许的情况下，可让 boom、stick、bucket、swing 在小偏差区间内各自覆盖一些 episode。
- 40 条有设计的多样化 episode，可能比 80 条几乎重复轨迹更有价值。

## 5. 建议验证实验

### 5.1 FPV 阴影归因实验

目的：

- 验证小沙丘阴影是否为 spurious cue。

实验：

1. 将训练 FPV 中的小沙丘阴影区域 mask 或抹平后喂给模型。
2. 将实时 FPV 人工加上类似训练画面的局部阴影后喂给模型。
3. 固定 qpos/qvel，比较 action 分布是否恢复或崩掉。

判断：

- 如果训练 FPV 去掉阴影后 action 崩，实时 FPV 加上阴影后 action 恢复，说明模型确实依赖该阴影。

### 5.2 统一预处理消融

目的：

- 判断 ROI/mask/blur/downsample 是否能减少训练 FPV 与实时 FPV 的差异。

实验组合：

- 原始训练 FPV + 原始实时 FPV。
- 预处理训练 FPV + 预处理实时 FPV。
- 不同 blur/downsample 强度。
- 不同 ROI/mask 方案。

关键指标：

- `实时FPV + 训练qpos/qvel` 是否恢复到接近训练 action 分布。
- action 是否仍能保留任务阶段变化，而不是变成过度平滑或无效输出。

### 5.3 异步控制链路实验

目的：

- 验证 control pump 和 policy worker 是否能提升真机控制稳定性。

指标：

- 控制发送 Hz。
- policy 更新 Hz。
- `policy_inference_latency_ms` P50/P95。
- action age P50/P95。
- read_state、send_action、日志写入耗时。
- watchdog 或 health fault 次数。

预期：

- 控制发送稳定在 50Hz。
- policy 更新率可以低于 50Hz，但 action age 不超过安全阈值。
- 真机行为比串行主循环更平稳。

## 6. 建议探索优先级

第一阶段：确认瓶颈和快速止血

1. 启用 control pump，使控制下发保持 50Hz。
2. 记录每步耗时拆分，确认 policy、read_state、log 的真实耗时。
3. 做 FPV 阴影归因实验。
4. 补录少量没有小沙丘阴影和小沙丘位置变化的 episode。

第二阶段：降低 FPV 局部特征敏感性

1. 实现训练和推理一致的 ROI/mask 或轻微 blur/downsample。
2. 加入训练时局部 shadow augmentation。
3. 对比冻结视觉 backbone 与正常训练。
4. 将数据补到 30 条左右，做第一轮重训和真机 shadow/control。

第三阶段：提升部署稳定性和数据覆盖

1. 实现异步 policy worker。
2. 按图像时间戳去重，避免重复帧完整推理。
3. 数据补到 40-60 条有设计的 episode。
4. 如果仍对 FPV 细节敏感，再补到 80-100 条，并增强土面、光照、起点扰动覆盖。

## 7. 当前推荐结论

综合现有证据，当前最可能的主因不是 gohome qpos 误差，而是 FPV 画面中的局部阴影和土面细节导致视觉输入分布偏移。速度问题则主要来自将 policy 完整推理绑定在 50Hz 主循环中。

最推荐的近期路线：

1. 控制链路上启用 50Hz control pump。
2. policy 侧接受低于 50Hz 的异步更新，并记录 action age。
3. 对 FPV 做训练和推理一致的轻量预处理，优先 ROI/mask 和轻微低通。
4. 训练时增加局部 shadow augmentation。
5. 补录覆盖“有阴影/无阴影/阴影位置变化”的反事实 episode。
6. 尝试冻结或半冻结视觉 backbone，降低小数据下的纹理记忆。

## 8. 补充思路

以下补充项不是推翻前面的判断，而是为了让后续排查和改进路线更完整。当前一号嫌疑仍然是实时 FPV 视觉分布偏移，速度问题仍然优先通过 control pump、异步更新和 profiling 解决。

### 8.1 control pump 需要配套 stale action 安全机制

control pump 能解决 50Hz heartbeat 和控制下发不断供的问题，但它不会自动保证 action 是新的。

建议补充：

- 记录并监控 `policy_action_age_ms`。
- 设置 `max_action_age_ms`，超过阈值后 guard 归零或平滑归零。
- 区分 `policy_update_hz`、`control_send_hz` 和 `camera_update_hz`。
- 如果 FPV、qpos/qvel 或 bridge state stale，不应继续无限重复上一帧速度命令。

原因：

- 对速度控制系统而言，重复旧 action 可能比降频本身更危险。
- pump 是控制下发层的止血方案，不应被误解为 policy 实时性问题已经完全解决。

### 8.2 profiling 需要包含 CUDA 同步和系统状态

当前建议记录 `read_state_ms`、`policy_ms`、`send_ms`、`log_ms` 是必要的，但 Jetson 上还应避免 CUDA 异步计时误判。

建议补充：

- policy 前向计时时在关键位置使用 `torch.cuda.synchronize()`。
- 单独统计 JPEG decode、numpy 到 torch、CPU 到 GPU copy、model forward、postprocess 的耗时。
- 记录 P50/P95/P99，而不仅是平均值。
- 同步记录 Jetson 电源模式、温度、GPU/CPU 频率和是否 thermal throttling。
- 测试 `torch.inference_mode()`、warmup、`model.eval()` 只调用一次时，需要在相同观测输入下对比 action diff。

原因：

- CUDA kernel 默认异步，Python wall time 可能不能准确代表真实前向耗时。
- Jetson 上温度、电源模式和频率限制可能让同一模型在不同测试轮次表现差很多。

### 8.3 做 pixel-level 的训练/实时图像路径一致性验证

文档中已经说明 RGB/BGR、shape、归一化、传输链路等问题已排除。为了让这个结论更可复现，建议增加 golden-frame replay。

建议实验：

1. 从训练 HDF5 中取一帧 FPV，走训练 dataset 读取路径。
2. 将同一帧构造成实时 observation，走 `PolicyActionSource` 和 `ACTAdapter.predict()` 的实时路径。
3. 在 ImageNet normalize 之后比较最终 tensor 的 `max_abs_diff`、`mean_abs_diff` 和 shape。
4. 对实时采到的一帧也做同样测试，确认 JPEG decode、resize、RGB 顺序和数值范围完全一致。

目标：

- 把“格式问题已排除”变成可重复脚本或固定检查项。
- 避免后续引入 ROI、blur、mask 后训练和推理路径再次漂移。

### 8.4 单独验证 action-label 时序和执行延迟

除了视觉 OOD，还应确认训练标签与实时执行之间的时间对齐没有系统偏差。

建议检查：

- 数据集中 `obs(t)` 对应的是 `action(t)`、`action(t-1)` 还是实际下发后延迟生效的 action。
- real dataset 当前对真实数据使用 `start = max(0, t0 - 1)`，需要确认这个补偿是否仍适合当前控制链路。
- 记录 `action_sample_timestamp_ns`、`action_send_timestamp_ns`、`controller_timestamp_ns`、`joint_timestamp_ns`，估计端到端延迟。
- 如有必要，训练时显式尝试不同 action offset，比较 shadow action 分布和真机行为。

原因：

- 行为克隆对时序错位很敏感。
- 如果视觉和动作标签有固定延迟，仅靠增加 FPV 增强和数据量可能无法完全解决闭环崩溃。

### 8.5 利用 ACT chunk 降低 query 频率

当前 `temporal_agg=true` 仍然每个 step 调一次模型。可以探索让模型以低于 50Hz 的频率输出 action chunk，再由控制层按 50Hz 执行或平滑执行。

可选方案：

- 每个新相机帧到达时 query 一次 policy。
- 每 N 个控制 step query 一次 policy，中间复用 chunk 中后续 action。
- 对 chunk action 增加低通或 max-delta guard，避免低频更新带来动作跳变。
- 与 control pump 结合，底层保持 50Hz，下层执行最新 chunk action。

优点：

- 更符合 ACT 本身一次预测一段动作的结构。
- 减少重复帧上的无效前向。
- 比强行让视觉模型 50Hz 前向更现实。

风险：

- 如果 chunk 内动作与真实闭环状态偏离，可能需要更短 query interval 或加入状态偏差 guard。

### 8.6 增加低视觉依赖 baseline

建议训练或测试一个弱 baseline，用来量化 FPV 依赖程度。

可选 baseline：

- qpos/qvel-only policy。
- 只使用更粗 ROI 或强低通 FPV 的 policy。
- 只预测任务阶段或动作方向的轻量 probe。

目标：

- 判断当前任务到底有多少信息必须来自 FPV。
- 当 FPV OOD 时，评估是否存在可降级的保守策略。
- 避免把所有失败都归因到视觉，而忽略低维状态或动作标签问题。

### 8.7 考虑更几何化的视觉表示

如果 RGB 对土面纹理和阴影始终过敏，可以考虑让模型看到更接近任务变量的输入。

可探索方向：

- 使用 depth 或 heightmap，弱化颜色、阴影和土体纹理。
- 对铲斗、土堆、工作区域做 segmentation 或 mask 后再输入。
- 使用边缘、粗深度、占用区域等几何特征作为辅助通道。
- 保留 RGB，但让 RGB 与 depth/segmentation 共同输入，降低模型只靠局部暗斑决策的机会。

注意：

- 新传感器或新表示会带来标定、同步和预处理一致性问题。
- 第一阶段仍应优先验证 ROI、blur、shadow augmentation 和反事实补录。

### 8.8 用纠偏数据而不只是成功轨迹补录

如果 shadow/control 已经能稳定复现某些失败状态，单纯补录更多相似成功 episode 可能效率不高。

建议：

- 在 policy 容易失败的实时 FPV 场景中，让 operator 接管并纠偏。
- 把失败前后的状态、人工纠偏动作、恢复轨迹加入训练集。
- 记录失败类型标签，例如无阴影、土面抹平、光照变化、gohome 偏差、action stale。
- 优先补 policy 当前会出错的状态分布，而不是只补重复的标准成功轨迹。

目标：

- 让数据覆盖真正的 closed-loop 分布。
- 降低小数据行为克隆在真机上越走越偏的问题。

### 8.9 做 occlusion 或 saliency 归因

阴影归因实验可以从手工 mask 小沙丘开始，但建议进一步系统化。

建议实验：

- 对 FPV 做 grid occlusion，逐块遮挡并观察 action 变化。
- 对训练 FPV 和实时 FPV 分别计算 action 对不同区域的敏感度。
- 比较敏感区域是否集中在小沙丘阴影、土体纹理、铲斗边缘或背景区域。
- 用归因结果指导 ROI/mask，而不是只靠人工观察选区域。

判断：

- 如果 action 对小沙丘阴影区域极敏感，说明当前 spurious cue 判断更可信。
- 如果 action 对铲斗边缘或土堆轮廓敏感，则 ROI/blur 不能过强，否则会损失关键任务信息。
