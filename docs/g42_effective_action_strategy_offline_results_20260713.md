# [Execution target G42/H1: 这轮到底做了什么]

这轮没有改真机运行时，也没有把分类器变成 gate。我们只在离线训练里改变了
“专家动作应该怎样教给 ACT”这件事：每个轴独立地把动作分成 neutral（不应动）、
positive、negative；低于实测机械死区的专家值在训练目标里记为 0；已经越过死区的
动作保留方向和幅度，并给刚越过边界的动作留 0.02 的余量。另加一个很轻的原始连续
动作损失，防止把操作者合法的多轴风格抹掉。

模型仍然只输出原来的连续四轴动作。三分类头只参加训练和诊断，运行时完全不拦截
第一次动作，不做 argmax，不压缩 action，不使用 joystick 的缩放来处理模型动作。

# [Execution target G42/H2: 为什么这样改]

已有数据给出了三个很明确的事实：

1. 真正有效的动作只占一部分，死区内的数值和“刚好越过死区”的数值在普通回归里
   会被当成相近的小数；但物理上前者是“不动”，后者是“必须动”。
2. 24 个正式 episode 的第一次有效动作 24/24 都是 bucket+。boom+ 的启动值尤其贴着
   边界：训练集启动比值中位数约为 1.050，validation 中位数约为 1.044，validation
   最小值只有 1.011。模型稍微保守一点，就会掉回死区。
3. 多轴动作不是非法模式。训练集里已经有大量 boom/bucket 联合动作，因此不能用
   “只允许一个轴”来修复问题。

所以本轮的原则是：把“停”和“动”的语义分开，同时继续模仿操作者的多轴连续风格；
不要在运行时用一个 gate 再次阻止 ACT。

# [Execution target G42/H3: 训练方案]

最终候选（H5）使用 20 个 train episode（包含 episode_77），固定 validation
`[94, 91, 84, 74, 92]`，没有读取或使用 held-out `105..109`。实测死区仍是：
swing `+0.661/-0.721`、boom `+0.259/-0.357`、stick `+0.5/-0.5`、bucket
`+0.408/-0.508`。

- 死区内目标：0。
- 有效动作：保留符号和相对幅度；边缘动作向外留 0.02 余量，最大值不超过 1。
- 切换后的前 4 tick 权重 4.0；持续有效动作权重 1.75；普通有效动作 1.5；neutral 1.0。
- 原始连续动作保留一条 0.35 权重的低强度监督，维护合法的同时多轴动作。
- neutral/positive/negative 分类只占很小的辅助损失（0.05），不参与推理决策。
- 从已经验证过的 G38 连续策略初始化，训练 2000 轮。

代码责任边界：死区语义在
`testbed/testbed/policies/act/effective_action.py`，数据集只生成目标和标签，
adapter 只组合损失，模型只增加辅助头；运行时动作源仍是连续 ACT。

# [Execution target G42/H4: 三个主要候选的结果]

下表均使用同一个 eye2 输入（video4/video5、qpos、identity action scale）和同一
24 episode 正式开放环测试。`hit-20` 是 20 tick 内越过目标方向死区的比例；state-hold
是 48 个冻结观测 anchor 的递归死锁测试。

| 方案 | 开放环 MAE | raw hit-20 | assist hit-20 | state-hold raw | state-hold assist | assist hidden | assist extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| eye2 baseline | 0.04728 | 96.79% | 98.69% | 35/48 | 43/48 | 3 | 11.27% |
| G38 A（边界损失） | 0.05158 | 96.93% | **99.27%** | 40/48 | **46/48** | 1 | 12.78% |
| H2：死区归零+分类（从 baseline） | 0.0552 | 96.14% | 98.92% | 39/48 | 42/48 | 2 | 7.86% |
| H4：再加 0.02 余量（从 A） | 0.0561 | 96.66% | 98.84% | 40/48 | 42/48 | 0 | 8.35% |
| **H5：余量+原始多轴风格（从 A）** | **0.0471** | **97.13%** | 98.97% | **41/48** | 43/48 | 2 | **5.90%** |

H5 的 raw 表现和额外动作都比旧方案好，但 mechanical assist 后仍只有 43/48，
没有超过 A 的 46/48；因此它没有通过完整离线 gate，不能替换当前参考方案。

最后做的 500 轮边界微调（从 H5 继续训练）验证了“再加同方向 hinge”并不会自动
解决问题：validation loss 从 H5 的 0.1054 变差到 0.1521，因此没有把这条微调当成
候选，也没有为它浪费 held-out 评估预算。

# [Execution target G42/H5: 额外动作到底是不是错误]

在 11,788 个专家有效帧上重新解释 extra：

- H5 raw 额外方向 314 个（2.66%），其中 294 个与专家原始动作同号且接近死区，
  312 个属于训练数据已经出现过的联合模式，只有 2 个没有找到数据支持。
- 加机械 assist 后额外方向 695 个（5.90%），只有 29 个（0.25%）没有专家支持。

这说明 H5 不是简单地“少动了”，而是确实减少了没有依据的额外轴动作；但它也说明
减少 extra 和保证每个边缘启动都成功之间存在真实取舍。

# [Execution target G42/H6: 完整离线检查结果]

H5 的完整报告在
`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260713/g42_effective_action_style_2000_formal/complete_offline_report/complete_offline_report.json`。

- 48 anchor：raw 41/48，assist 43/48；assist deadlock 5，hidden 2，方向反转 0。
- startup：raw 和 assist 都是 5/5；问题主要发生在中途的 boom+ 边缘切换。
- policy 输出越界 0 次，非有限值 0 次，单 tick 最多同时有效 2 个轴。
- tail extra 87 帧，release extra 317 帧；这些是开放环的诊断统计，不是现场液压
  故障标签。
- gohome 不能从正式 HDF5 判断：24 个 episode 都没有完整 gohome/tail handoff 标签。
- 已有执行响应 sidecar 的 validation 48/48 都观察到响应，但 sidecar 是 teleop
  启发式响应，不含 policy intent 和人工纠错，不能据此训练可靠 retry。
- held-out `105..109` 没有评估。

完整 gate 的结果是 `passed=false`，原因很具体：要求 assist 至少 46/48 且 hidden=0，
H5 只有 43/48 且 hidden=2。

# [Execution target G42/H7: 失败原因，用大白话说]

这轮已经排除了 action_scale 和运行时 gate 作为主要原因。剩下的失败样本不是“模型
完全不会 bucket/boom”，而是：在某些视觉状态下，专家的主轴动作刚好贴着死区边缘，
模型能看出大致阶段，却不能稳定判断“这一次必须坚定地启动 boom+”。冻结观测后，它
有时会把 boom 输出留在 0 附近；有时又把另一个轴输出得更强。分类标签减少了死区
混淆，却没有增加足够的时序/视觉证据来解决这个选择问题。

保留原始连续监督后，多轴风格被保住了，所以 H5 的 MAE、raw liveness 和 extra 都
改善；但这也证明“死区归零+分类”本身不是根治，根治需要让模型看到启动前后的短时
视觉变化，而不是只看一张当前图和 qpos。

# [Execution target G42/H8: 当前应该怎么用、下一步怎么做]

当前不替换 A，也不把 H5 部署到真机。A 的 assist state-hold 仍是最强的已验证参考，
但 A 自己还有 1 个 hidden anchor，仍不应被描述成已经解决现场问题。

下一步最值得做的不是再扫死区阈值，而是利用现有 HDF5 生成短时序输入：给 ACT 同时
看当前帧和前 2–4 帧（或等价的图像差分/短时视觉特征），让它区分“正在准备启动”和
“已经停住”。保留 H5 的有效动作目标、原始多轴 tie-breaker 和 transition weighting，
只把“启动时序证据”作为新的输入/辅助任务。这样仍然不需要新采集数据，也不需要
运行时 hard gate；如果下一轮 state-hold 不能达到至少 A 的 46/48 并把 hidden 压到 0，
就继续保留 A 作为离线参考，不进入现场。

# [Execution target G42/H9: 可复核产物]

- H5 config：`testbed/testbed/configs/act_real_gmsl_eye2_g42_effective_action_style_2000.yaml`，SHA256 `b512cfdc15756242c7cbca183b4bce6de6d25f28f1a6b5742563c103cb3f0208`
- H5 checkpoint：`.../g42_effective_action_style_2000_formal/ckpt/policy_best.ckpt`，SHA256 `22f7a595e60b72cef5563239e42ae6e8f279e52a4c9e84118029457f4a754ff6`
- H5 open-loop summary：`.../g42_effective_action_style_2000_formal/open_loop_all24/collection_summary.json`，SHA256 `c7dbc4ee9887e1cdb560b62a31174f1112926ae7cbc1a329be871fe539c65201`
- H5 state-hold summary：`.../g42_effective_action_style_2000_formal/state_hold_raw_assist_val5_h20/run_summary.json`，SHA256 `08a42fae8c028b4c14ce84a8c7b024042eea800c27fb622f80bf6da283ff4d27`
- H5 expert-supported extra：`.../g42_effective_action_style_2000_formal/expert_supported_extra.json`，SHA256 `77cba5b5578eced6a034b395640fb35fba2cbe039b510bf554a4f341fb84d897`
- H5 complete report：`.../g42_effective_action_style_2000_formal/complete_offline_report/complete_offline_report.json`，SHA256 `4f2a9a278c0575a32a2cddca53ba59e44c85a3427968411b290d5afbd5123ee8`
- 固定 split：`.../runs/goal_state_liveness_20260712/g37_transition_aware_formal/train_val_split.yaml`，SHA256 `14ab7e9e67382646bc5b922ac48ab31eecd41217caa9366f0259f85a0c2844f6`
