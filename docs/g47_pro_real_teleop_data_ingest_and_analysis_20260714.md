# [Execution target G47/H1: 数据已经安全复制到哪里]

外接盘当前设备是 `/dev/sdb1`。原来的 `/mnt/external_usb` 指向已经消失的
`/dev/sda1`，读取会报 I/O error；本轮先卸载失效挂载，再把当前设备以只读方式
挂载到 `/mnt/pro_real_teleop_usb`，没有修改源盘。

源目录：

`/mnt/pro_real_teleop_usb/pro_real_teleop`

本地原始副本：

`/data/pingfan/Excavator_real_stack_data/pro_real_teleop_20260713_usb_raw`

复制先进入 `.partial` 目录，190 个文件逐个做 SHA-256，对比完全一致后才原子改名。
源和目标都是 47,715,104,244 bytes。源目录包含 170 条顶层成功录制、18 个 failed
HDF5 和 2 个失败临时文件；失败文件也完整保留，但不进入训练。

复制清单在
`/data/pingfan/Excavator_real_stack_data/runs/g47_pro_real_teleop_ingest_20260714/ingest_manifest.json`。
文件哈希清单 SHA-256 是
`172cff65d458a1228febf901c48c81e3ba07554ffb3a998e948d18fc8978ff05`。

# [Execution target G47/H2: 哪些录制属于同一个正式协议]

170 条顶层成功录制不是一个完全相同的录制协议：

- `episode_0` 是第一版手柄参数试验，swing 方向和后续不一致，response profile 未启用；
- `episode_1` 是第二版短试验；
- `episode_2..175` 中实际存在的 168 条共享同一套正式参数：四轴 scale
  `[0.8, 0.8, 0.8, 0.7]`，response profile 启用，方向与后续录制一致。

这里的 scale 是录制操作者指令时的手柄响应整形，不是 policy runtime 的
`action_scale`。在正式人工操作段中，HDF5 的 `/action` 与
`diagnostics/raw_action` 逐元素完全相同；也就是说训练目标确实是经过手柄响应曲线
形成的操作者命令，没有再次把模型动作缩放的问题。

完整参数分组和动作链证据在
`/data/pingfan/Excavator_real_stack_data/runs/g47_pro_real_teleop_ingest_20260714/recording_contract_audit.json`，
SHA-256 `9060f3993f3844ca86b6e6074ab459f87bc07fe262bb88cd9b89cb20fd45c821`。

# [Execution target G47/H3: 20Hz 数据和完整 QC 结果]

正式协议的 168 条数据已经生成不修改原始 HDF5 的 20Hz 副本：

`/data/pingfan/Excavator_real_stack_data/pro_real_teleop_20260713_20hz_v1`

使用当前仓库既有协议：20Hz、action label offset `-0.02s`、250ms gap mask、前后
各 1s padding。结果是 98,467 steps，人工任务段约 1.365 小时，只有 1 条 episode
出现超过 250ms 的源时间 gap，共 mask 41 steps。

基础 HDF5 QC 对原始 170 条全部通过：均可读、有四相机、qpos/qvel/action 均为四轴、
时间戳单调、shape 一致、无 NaN/Inf、real metadata 和单位正确。

通用 training QC 在 20Hz 数据上给出 160 PASS、8 FAIL，但逐条复核后不能照单全收：

- `episode_124/126/146` 全程 action 为零、机械状态基本静止，是真正的误录，排除；
- `episode_2/6/7/32` 是较慢但完整的挖掘循环，长度不是故障，重新纳入；
- `episode_36` 的 bucket 相邻采样变化 0.206rad，但两帧间隔实际为 87ms，期间 bucket
  保持约 0.63 的有效命令，qvel 也同方向；这是物理快速动作，不是传感器分支跳变，
  重新纳入。

因此正式协议 168 条中，165 条经过复核可用。reviewed manifest 在
`/data/pingfan/Excavator_real_stack_data/runs/g47_pro_real_teleop_ingest_20260714/reviewed_train_ready_manifest.json`，
SHA-256 `21a46d500b125f4d800ad15c271f248f3e818fb08cf7aa4f83fecbe68890c494`。

# [Execution target G47/H4: 新数据相对旧数据改变了什么]

对比使用旧正式 24 条和新描述集 160 条。新描述集暂时排除 `124/126/146`，并把新
数据中数字编号恰好也是 `105..109` 的五条单独保留，避免与旧冻结 held-out 发生
任何工具或人工混淆。

| 指标 | 旧风格 | 新风格 |
|---|---:|---:|
| episode | 24 | 160 |
| 20Hz steps | 16,529 | 94,853 |
| swing 有效动作占比 | 27.10% | 41.02% |
| boom 有效动作占比 | 19.02% | 35.03% |
| stick 有效动作占比 | 0% | 11.67% |
| bucket 有效动作占比 | 32.88% | 33.44% |
| 同时两轴有效 | 7.68% | 26.72% |
| 同时三轴有效 | 0% | 4.38% |
| 同时四轴有效 | 0% | 0.24% |

这说明新数据不是简单“多了一些相同演示”：它真正补齐了 stick、更多 boom、更多
多轴耦合以及更丰富的挖掘点位。任何只允许单轴、把额外轴一律视为错误、或者用 hard
gate 选择唯一轴的训练/运行时方案，都与这批专家数据直接冲突。

完整对比在
`/data/pingfan/Excavator_real_stack_data/runs/g47_pro_real_teleop_ingest_20260714/old_new_action_distribution_comparison.json`，
SHA-256 `a1a840936f6c81d2e5360eb8a0733c00275ad959e0e4ea85bce0c8cf692e79a7`。

# [Execution target G47/H5: 启动动作和死区问题有没有被新数据改变]

旧风格 24/24 的第一有效动作都是单独 `bucket+`。新风格的第一有效动作更依赖初始
姿态和目标点：

- 单独 `stick+`：112/160；
- 单独 `boom-`：39/160；
- swing：4/160；
- 同一 tick 多轴启动：5/160。

这不表示方向冲突。用户已经确认相似阶段方向一致；这里反映的是新数据包含更多不同
初始几何和接近挖掘点的过程。因此不能再用“启动永远是 bucket”作为测试定义，必须按
每条 episode 自己的第一段专家有效主轴和后续持续响应来测。

更关键的是，数据量变大并没有让死区边界自动消失：

- 旧数据第一启动幅度/死区比值 P10 为 1.0077，中位数 1.0311；
- 新数据 P10 为 1.0080，中位数 1.0355；
- 新数据中 stick+ 有 1,647 steps 位于死区的 90%-100%，刚越过死区的
  100%-110% 又有 1,857 steps；stick- 对应为 3,241 和 2,198 steps。

所以“死区内专家 action 归零 + 每轴 neutral/positive/negative 辅助监督”仍然有明确
物理依据。与旧 G42 相比，新数据给它提供了真正的 stick 正负样本和约六倍总 steps，
更可能学到什么时候启动。但新数据有 31.3% 的帧是两轴或更多轴同时有效，因此必须
同时保留原始连续动作监督；只做归零和分类、抹掉连续多轴风格的版本不应再作为主候选。

精确 startup combo 报告在
`/data/pingfan/Excavator_real_stack_data/runs/g47_pro_real_teleop_ingest_20260714/startup_exact_combo_comparison.json`，
SHA-256 `212f49bef4f2a92fde110df804c009bc42ae0a904bf18ae2e7b674c914438ecf`。

# [Execution target G47/H6: 新旧视觉能不能直接混在一起]

对旧、新各均衡抽取 24 条 episode，每条 24 帧，只看 policy 当前使用的 video4/video5，
做了同一特征空间的 8 类外观聚类和 contact sheets。结果不是“完全相同的场景”：

- domain 1 和 6 两边都有较多样本，说明有共享视觉区域；
- domain 0 和 7 全是新数据，主要是新点位、后期沙面和不同机械构型；
- domain 5 有 94.1% 来自旧数据，代表旧固定场景/阶段在新数据中比例很小。

因此新数据正好补上了现场分布覆盖，是这批数据最有价值的部分。它也说明训练后应以
新风格 validation/test 为准，不能用旧固定场景得分代替。旧数据可以帮助视觉和任务
阶段预训练，但最终应在新数据上强化；否则模型可能在旧风格的固定视角和轨迹上看起来
很好，进入新点位时仍然失去动作信心。

这个聚类只用手工外观特征，仍会混入任务阶段，不能当真实语义标签。产物和 contact
sheets 在
`/data/pingfan/Excavator_real_stack_data/runs/g47_pro_real_teleop_ingest_20260714/cross_style_visual_eye2_k8`；
对比 JSON SHA-256 `14e2ce07e47208695912ee1c0f06b7d9721a3551a53bb1ae38d37f3234dcf966`。

# [Execution target G47/H7: 冻结划分与数据泄漏边界]

这批数据都来自同一个 `session_id=pro_real_teleop`，所以不能声称真正做到
session-disjoint。为了比随机逐 episode 切分更严格，本轮在任何模型训练前冻结了连续
时间块：

- train：前 120 条可用 episode；
- validation：`135..155` 中实际可用的 20 条；
- test：`156..175` 共 20 条，只允许候选、配置、checkpoint 全部冻结后评估；
- 新数据的 `105..109` 暂时保留，不进入这轮模型池，避免与旧 held-out 的同名编号
  混淆；旧数据集的 `episode_105..109` 继续严格禁止训练、校准和选择。

split 在
`/data/pingfan/Excavator_real_stack_data/runs/g47_pro_real_teleop_ingest_20260714/g47_new_style_chronological_split_v1.json`，
SHA-256 `e52f76853df2fa260646a630b6cea3497efd71681b66e6fb8be2afe15de97d45`。

# [Execution target G47/H8: 下一轮训练应该怎样比较]

第一轮不要同时改数据、输入历史、temporal ensemble 和运行时模块，否则无法知道提升
来自哪里。建议固定上述新 validation/test，先训练以下可归因对照：

1. 新数据 train-only 的普通 continuous ACT，作为新的真实 baseline；
2. 旧正式 train + 新 train 的普通 continuous ACT，检验单纯增加数据是否提升；
3. 同一混合数据上加入死区内归零、每轴三分类、启动/持续动作加权，同时保留低权重
   原始连续 action；
4. 用全部兼容 train 预训练，再只用新 train 低学习率强化，作为主候选。

四组都使用相同输入、identity policy action scale、2000 epoch 预算和同一验证协议。
选择标准不能只看 MAE，必须包含：每条 episode 的真实主轴 startup、所有 transition、
逐轴死区命中、wrong/opposite、多轴支持、release/tail、recursive state-hold、raw 与
mechanical assist 两种结果。只有前三组证明问题主要剩在可观测性时，才把可回退的
2-4 帧短时输入作为下一独立变量。

当前结论是：新数据显著增加了 ACT 真正需要的状态和动作覆盖；最有依据的训练路线是
“新数据主导 + 死区语义辅助 + 保留连续多轴动作”，而不是机械 assist、hard gate 或
单轴分类控制。此文档只完成数据落盘和分析，没有训练 checkpoint，也没有读取冻结
test 上的任何模型结果。
