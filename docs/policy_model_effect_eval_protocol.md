# Policy 模型效果评估协议

本文档固定以后比较 policy 模型效果时的离线分析口径。整体 MAE 只能说明
全局平均误差，不能单独判断局部动作是否能开始、是否能正确结束，也不能判断
模型是否只是输出接近 0 的安全动作。因此模型效果必须按本文方法分析。

## 结论口径

1. 不用整体 MAE 单独决定模型好坏。MAE 只作为全集回归指标。
2. 局部窗口必须看逐步 `policy_action`，尤其是起步窗口、任务动作窗口和结束/回收窗口。
   “有效动作”必须按每个关节正/负方向的液压死区判断，不能再用固定全局阈值代替。
3. 现场或 shadow 日志中要优先区分原始 `policy_action` 与 `safe_action`、
   `commanded_action`。模型能力判断看原始 `policy_action`；安全层裁剪或 shadow zero
   只能用于解释实际下发链路。
4. 图和 CSV 必须同时保留：`action_timeseries.png` 用于看动作形态，
   `actions.csv` 用于统计局部窗口的幅值、方向和持续时间。
5. 使用视觉输入的模型必须做 FPV 敏感性检查，不能只做同一 episode 的正常 replay。
6. 每次训练完成后的离线 eval 必须包含固定 qpos、多 FPV 替换 replay。这个检查不是
   live 前临时加测，而是训练 checkpoint 是否可进入 shadow/live 的固定 gate。

## 现有可用工具

主工具是 `testbed/testbed/cli/offline_policy_eval.py`，推荐用 repo-local module
方式运行：

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
MPLCONFIGDIR=/tmp/excavator_mpl \
python -m testbed.cli.offline_policy_eval ...
```

单 episode replay 会调用 `evaluate_episode()`，每一步使用 HDF5 中记录的
`observations/qpos`、`observations/qvel` 和 FPV 图片构造观测，输出
`policy_action`，再与 HDF5 的 expert `action` 对比。这是 open-loop replay，
不会把 policy 动作回灌到机器状态里。

正式模型效果评估必须跑完整 episode，再从 `actions.csv` 截取局部窗口。
`--max-steps` 只能用于快速 smoke test，不能作为正式起步窗口结论来源；因为当前
CLI 会用截断后的 episode 长度初始化 policy replay，可能影响 ACT temporal
aggregation 的早期输出。

单 episode 输出：

- `summary.json`：整体和分轴 MAE/RMSE/p95/max 等指标。
- `actions.csv`：逐步 `expert_*`、`policy_*`、`error_*`，这是局部动作分析的主证据。
- `actions.npz`：同样的逐步动作数组，便于脚本分析。
- `action_timeseries.png`：动作曲线，必须用于人工检查局部形态。
- `action_distribution.png`：动作分布。

全集输出通过 `--all-train-ready` 产生：

- `collection_summary.json`
- `episode_metrics.csv`
- `collection_action_distribution.png`
- `episode_policy_vs_expert_p95.png`
- `episode_mae_by_axis.png`

全集输出只能用于筛查整体回归，不替代局部窗口判断。

## 液压死区标定

有效动作的定义：

> 对某一关节某一方向，`policy_action` 的符号与目标/专家动作方向一致，并且
> 绝对值达到该关节该方向的液压死区阈值之外，才算该轴在这一帧输出了有效动作。

死区阈值必须分轴、分方向记录：

| 轴 | 正方向死区 | 负方向死区 |
| --- | --- | --- |
| swing | `swing.pos` | `swing.neg` |
| boom | `boom.pos` | `boom.neg` |
| stick | `stick.pos` | `stick.neg` |
| bucket | `bucket.pos` | `bucket.neg` |

阈值用 action 的归一化命令幅值表示，例如 `bucket.pos = 0.07` 表示
`policy_bucket >= +0.07` 才算 bucket 正方向有效；`bucket.neg = 0.06` 表示
`policy_bucket <= -0.06` 才算 bucket 负方向有效。

### 标定来源优先级

1. 优先使用真机单轴响应标定结果。现有工具是 `scripts/calibrate_axis_response.py`，
   它会逐轴、逐方向发送不同幅值，并打印 first responsive command。
2. 如果没有最新单轴标定，使用录制 HDF5 中的 `action`、`observations/qpos` 和
   `observations/qvel` 离线估计死区。
3. 如果某个轴/方向数据不足，默认死区取 `0.5`。报告中必须标明该值来自
   `default_insufficient_data`，后续真机单轴标定或足够 HDF5 样本可以覆盖它。
4. 没有死区表时，不允许把“动作有效/无效”写成确定结论；只能写“按临时幅值阈值观察”。

真机单轴标定命令模板：

```bash
python3 scripts/calibrate_axis_response.py \
  --host 127.0.0.1 \
  --port 8766 \
  --axis bucket \
  --direction both \
  --amplitudes 0.03,0.05,0.07,0.10,0.12 \
  --duration-s 0.45 \
  --settle-s 0.80 \
  --abort-delta-rad 0.05 \
  --confirm-hardware-motion
```

### HDF5 离线估计方法

从录制数据估计死区时，不能只看单帧 `action` 和同帧 `qpos`，因为液压响应有延迟。
固定方法如下：

1. 对每个 train-ready episode 读取 `action[t, axis]`、`qpos[t, axis]` 和
   `qvel[t, axis]`。
2. 对每个轴、每个方向分别取样，只保留该方向 action 连续保持同符号的片段。
3. 对每个候选幅值 bin，检查 action 发出后 `1..K` 步内的响应，默认 `K=4`
   个 20Hz step，即约 0.2s。
4. 响应成立条件：
   - `qpos[t+lag] - qpos[t]` 的符号与该方向一致，且绝对值超过 qpos 噪声阈值；或
   - `qvel[t+lag]` 的符号与该方向一致，且绝对值超过 qvel 噪声阈值。
5. 每个候选幅值统计响应率、同向响应中位数、反向/误轴响应比例。
6. 死区阈值取“最小稳定响应幅值”：响应率达到要求，且同向响应中位数稳定为正，
   同时反向响应比例低。

建议默认门槛：

- 响应窗口：`K=4`。
- qpos 响应阈值：使用静止/近零 action 段估计噪声，取 `max(3 * qpos_noise_std, 0.0015 rad)`。
- qvel 响应阈值：使用静止/近零 action 段估计噪声，取 `max(3 * qvel_noise_std, 0.006 rad/s)`。
- 稳定响应率：至少 70%。
- 每个轴每个方向的有效样本数不足时，该方向死区使用默认值 `0.5`，并标为
  `default_insufficient_data`，不能写成数据估计值。

### 死区表交付格式

每次模型对比必须引用一个死区表，并写清来源：

```json
{
  "source": "axis_response_calibration | hdf5_action_qpos_estimate",
  "dataset_or_artifact": "<path>",
  "control_hz": 20,
  "deadzone_action": {
    "swing": {"pos": 0.0, "neg": 0.0},
    "boom": {"pos": 0.0, "neg": 0.0},
    "stick": {"pos": 0.5, "neg": 0.5},
    "bucket": {"pos": 0.0, "neg": 0.0}
  }
}
```

`0.0` 只是占位，不允许作为真实死区结论提交；数据不足轴/方向默认使用 `0.5`。

## 必跑分析项

每次对比两个或多个模型时，必须保证 dataset、manifest、episode、窗口长度、
checkpoint 选择和 temporal aggregation 设置一致。

### 0. 训练后固定 qpos / 多 FPV gate

每次新训练产出 checkpoint bundle 后，必须先跑固定 qpos、多 FPV 替换 replay，再决定
是否进入 shadow 或 live control。这个 gate 用来检查模型是否把土堆纹理、阴影、光照、
相机微偏或 dump 后画面细节当成动作阶段信号。

固定原则：

- qpos/action 轨迹来自目标 episode，保持不变。
- FPV 图像来自多个 episode 或 held-out/live-like 画面。
- 使用 `--image-step-mode nearest_qpos` 优先匹配相近姿态；如果匹配质量差，这个
  FPV 替换结果只能标记为 `inconclusive`，不能当作 pass。
- 输出必须按起步窗口、任务动作窗口、结束/回收窗口分别统计，不允许只看整体 MAE。

推荐最小覆盖：

| 类别 | 目的 |
| --- | --- |
| 训练集内 qpos + 同 episode FPV | 确认正常 replay 没退化 |
| 训练集内 qpos + early/late 其它 FPV | 检查视角子域变化 |
| return-like qpos + live-like 或 held-out FPV | 检查 return 动作是否被画面削弱 |
| terminal/end qpos + live-like 或 held-out FPV | 检查尾段是否被画面诱发多动 |

命令模板：

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
MPLCONFIGDIR=/tmp/excavator_mpl \
python -m testbed.cli.offline_policy_eval \
  --bundle-dir <MODEL_BUNDLE_DIR> \
  --dataset-dir /data/pingfan/Excavator_real_stack_data/external_usb_datasets/real_teleop_v1_repaired_20hz_v1 \
  --episode-id <QPOS_ACTION_EPISODE> \
  --image-episode-id <FPV_EPISODE> \
  --image-step-mode nearest_qpos \
  --output-dir <OUTPUT_DIR>
```

每个 `<OUTPUT_DIR>` 必须保留：

- `summary.json`
- `actions.csv`
- `actions.npz`
- `action_timeseries.png`
- `action_distribution.png`

固定 qpos / 多 FPV gate 的报告表至少包含：

| 字段 | 含义 |
| --- | --- |
| `model_bundle` | checkpoint bundle |
| `qpos_action_episode` | 提供 qpos/action 的 episode |
| `fpv_episode` | 提供 FPV 的 episode |
| `image_step_mode` | 通常为 `nearest_qpos` |
| `image_match_p95` | `summary.json` 中的 qpos 匹配质量指标 |
| `window` | `start40`、主动作段、`end80` 等 |
| `policy_any_effective_pct` | 按死区表计算的有效动作比例 |
| `same_axis_dir_effective_pct` | expert 有效时同轴同向比例 |
| `extra_or_wrong_pct` | expert 不需要该轴/方向时的额外或错误有效动作 |
| `decision` | `pass`、`fail` 或 `inconclusive` |

判定口径：

- 如果正常 replay 通过，但替换 FPV 后关键窗口动作低于死区或方向明显改变，模型不能进入
  control，只能继续 shadow 或回到数据/训练改进。
- 如果替换 FPV 后尾段 `extra_or_wrong_pct` 明显高于当前 baseline，同样不能进入 control。
- 如果 `image_match_metrics` 显示 qpos 匹配很差，本次结果记为 `inconclusive`，需要换
  episode 或改用更合适的 image mapping。
- 这个 gate 通过不代表模型一定能 live 成功；它只是 live 前必须满足的最低离线条件。

### 1. 起步窗口

用于判断模型在 home pose 或接近 home pose 时是否能给出有效启动动作。

命令模板：

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
MPLCONFIGDIR=/tmp/excavator_mpl \
python -m testbed.cli.offline_policy_eval \
  --bundle-dir <MODEL_BUNDLE_DIR> \
  --dataset-dir /data/pingfan/Excavator_real_stack_data/external_usb_datasets/real_teleop_v1_repaired_20hz_v1 \
  --manifest data/qpos_only_31_eval_20260611/train_ready_manifest_31_no66.json \
  --episode-id <EPISODE_ID> \
  --output-dir <OUTPUT_DIR>
```

必须记录：

- 从完整 `actions.csv` 截取的起步 step 范围，例如 `[0, 40)`；不要用
  `--max-steps 40` 的输出作为正式结果。
- `summary.json` 中的 `metrics.overall.policy_p95_abs` 和
  `metrics.overall.policy_max_abs`。
- `actions.csv` 中每一步四轴 `policy_*` 的最大绝对值。
- 每个轴按对应方向死区判断的有效帧比例，例如
  `policy_bucket >= bucket.pos_deadzone` 或 `policy_bucket <= -bucket.neg_deadzone`。
- 如果暂时保留 `max_abs < 0.05` 和 `max_abs < 0.10`，只能作为辅助观察项，
  不能用于最终“有效动作”结论。
- `policy_swing`、`policy_boom`、`policy_stick`、`policy_bucket` 的方向和持续性。
- `action_timeseries.png` 路径。

示例局部统计。假设已经把死区写成 shell 变量：

```bash
BUCKET_POS_DZ=0.07
awk -F, -v dz="$BUCKET_POS_DZ" 'NR>1 {
  n++
  if($10 >= dz) effective++
}
END {
  printf "n=%d bucket_pos_effective_pct=%.2f\n", n, 100*effective/n
}' <OUTPUT_DIR>/actions.csv
```

### 2. 任务动作窗口

用于判断模型是否在真实挖掘动作段输出正确轴、正确方向和足够幅值。先跑完整
episode，不加 `--max-steps`，再从 `actions.csv` 中截取 expert 动作明显非零的局部段。

必须记录：

- 选取的 step 范围和选择原因。
- expert 与 policy 在四轴上的方向是否一致。
- policy 是否出现错误轴的明显动作。
- 局部段 `policy_p95_abs`、`policy_max_abs`、平均绝对值。
- 局部段按死区表计算的有效帧比例。对 expert 要求非零的轴，policy 必须越过
  对应方向死区才算动作有效。
- 曲线图中 policy 是否跟随 expert 的动作阶段变化。

### 3. 结束/回收窗口

用于判断动作是否能正确停下、回收或回到接近 home pose，而不是只会启动。
当前现有 CLI 没有 `--start-step`，所以结束窗口必须先跑完整 episode，再从
`actions.csv` 的末段或回收阶段截取局部窗口。

必须记录：

- 结束窗口 step 范围。
- policy 是否在 expert 停止后仍持续输出大动作。
- policy 是否在需要回收时输出接近 0。
- 各轴末段 `policy_max_abs` 和 `policy_p95_abs`。
- 如果结束/回收阶段需要某轴反向回收，policy 必须越过该轴反向死区；如果目标是停止，
  policy 应回到正/负死区以内，不能持续越过死区。
- `action_timeseries.png` 中结束阶段截图或路径。

### 4. FPV 敏感性

视觉模型必须检查“同一 qpos/action，替换 FPV 后 policy 是否改变”。这能区分
模型是否真的使用图像，以及图像变化是否导致异常动作。

命令模板：

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
MPLCONFIGDIR=/tmp/excavator_mpl \
python -m testbed.cli.offline_policy_eval \
  --bundle-dir <MODEL_BUNDLE_DIR> \
  --dataset-dir /data/pingfan/Excavator_real_stack_data/external_usb_datasets/real_teleop_v1_repaired_20hz_v1 \
  --manifest <MANIFEST> \
  --episode-id <QPOS_ACTION_EPISODE> \
  --image-episode-id <FPV_EPISODE> \
  --image-step-mode nearest_qpos \
  --output-dir <OUTPUT_DIR>
```

必须检查 `summary.json` 中的 `image_match_metrics`，尤其是
`p95_max_abs_delta_rad` 和 `unique_image_steps`。如果 qpos 匹配很差，这次
FPV 替换结果不能直接当成模型视觉敏感性结论。

同一 episode 内还可以用 `--image-transform` 做确定性的 FPV 变换回测，用来判断
模型是否对视角缩放、中心裁剪或图像低通过敏：

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
MPLCONFIGDIR=/tmp/excavator_mpl \
python -m testbed.cli.offline_policy_eval \
  --bundle-dir <MODEL_BUNDLE_DIR> \
  --dataset-dir /data/pingfan/Excavator_real_stack_data/external_usb_datasets/real_teleop_v1_repaired_20hz_v1 \
  --manifest <MANIFEST> \
  --all-train-ready \
  --image-transform center_zoom_085 \
  --output-dir <OUTPUT_DIR>
```

当前支持的变换包括 `none`、`center_zoom_085`、`center_zoom_075`、
`center_zoom_085_blur`、`center_zoom_075_blur`、`downsample_080` 和
`downsample_060`。这些变换会保持输出图片尺寸和 RGB 三通道不变。

注意：只在离线 replay 时加 `--image-transform` 只能用于筛查方向，不能直接证明
“processed FPV 模型”效果更好。若某个变换让动作更稳定，下一步必须用同一个
transform 重新训练或微调，再按起步、任务动作、结束/回收窗口和 FPV 替换 replay
完整评估。

训练期也可以在配置的 `train.image_transform` 中写入同名 transform。训练 dataset
和离线 replay 共用同一套实现，因此可以做“原图训练”和“processed FPV 训练”的
可比实验；但上真机前还必须让实时推理链路使用同样的图像预处理。

### 5. Temporal aggregation 对照

ACT 默认使用 temporal aggregation。若局部动作异常，要追加一次
`--no-temporal-agg`，判断问题来自模型原始 chunk，还是 temporal aggregation
平滑后形成的输出。

## 全集回归

全集回归用于确认模型没有整体退化，但不能作为唯一结论。

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
MPLCONFIGDIR=/tmp/excavator_mpl \
python -m testbed.cli.offline_policy_eval \
  --bundle-dir <MODEL_BUNDLE_DIR> \
  --dataset-dir /data/pingfan/Excavator_real_stack_data/external_usb_datasets/real_teleop_v1_repaired_20hz_v1 \
  --manifest <MANIFEST> \
  --all-train-ready \
  --output-dir <OUTPUT_DIR>
```

报告中可以引用全集 MAE、分轴 MAE、episode p95 分布，但最终模型选择必须同时满足
起步、任务动作、结束/回收窗口的局部检查。

## 已有样例

同样使用 episode 66 起步 40 步窗口时，已有输出说明 MAE 会误导判断：

- `runs/offline_policy_eval/rerun_curve_fpv000_ep66_start40_20260611`
  - `mae = 0.0077`
  - `expert_p95_abs = 0.0000`
  - `policy_p95_abs = 0.0124`
  - `policy_max_abs = 0.0149`
  - `max_abs < 0.05` 的帧比例为 100%
- `runs/offline_policy_eval/rerun_curve_fpv025_ep66_start40_20260611`
  - `mae = 0.0878`
  - `expert_p95_abs = 0.0000`
  - `policy_p95_abs = 0.3439`
  - `policy_max_abs = 0.5159`
  - `max_abs < 0.05` 的帧比例为 5%

这说明 fpv000 在该起步窗口几乎不动，反而因为 expert 前 40 步也为 0 得到更低
MAE；fpv025 虽然 MAE 更高，但局部曲线显示 bucket 轴已经输出明显启动动作。
下一步判断 fpv025 的 bucket 动作是否“有效”，必须把 `policy_bucket` 与
bucket 正方向死区比较；只有越过死区的帧才算有效启动帧。因此模型效果结论必须写成
“死区标定 + 局部动作能力 + 全集回归”的组合结论，不能只写 MAE。

## 最低交付清单

一次模型效果对比至少交付以下内容：

1. 每个模型的起步窗口 `summary.json`、`actions.csv`、`action_timeseries.png`。
2. 本次使用的每轴正/负方向液压死区表，以及死区表来源。
3. 每个模型的任务动作窗口 step 范围和局部统计。
4. 每个模型的结束/回收窗口 step 范围和局部统计。
5. 每个局部窗口按死区表计算的有效动作比例。
6. 如果模型使用 FPV，交付 FPV 替换 replay 结果和 `image_match_metrics`。
7. 全集 `collection_summary.json` 与 `episode_metrics.csv`。
8. 结论明确区分：局部动作是否越过死区、全集 MAE 是否退化、是否存在现场风险。
