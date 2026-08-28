# 左右目标脚本 ACT 真机运行说明

## 适用范围

本流程运行已经通过离线验收的 `policy_accepted.ckpt`。脚本规划器为每一铲提交左区或
右区目标，ACT 完成整铲动作；真实 qpos/qvel 在目标区域连续稳定 0.5 秒后，系统才提交
下一铲目标。

本流程不包含精确位置控制。左、右目标仍使用现有 ready 区域定义。

## 运行链

```text
人工模式稳定在脚本初始区域
→ 按按钮 4 请求模型控制
→ 运行时提交第一铲目标并清空 ACT 历史
→ ACT 执行挖掘、卸料和回程
→ 必须先观察到一次真实回转位移
→ 训练数据支持的目标区域内连续稳定 0.5 秒
→ 结束当前一铲并提交下一目标
→ 脚本结束、错误区域或超时：锁定零动作
→ 再按一次按钮 4：确认锁零并留在人工模式
```

同侧连续目标也必须先完成真实回转位移，不会因为起点已经位于目标区域而跳过一铲。

## 文件

- 默认安全配置：`testbed/testbed/configs/policy_real_transition_target_release_v2.yaml`
- 示例脚本：`testbed/testbed/configs/real_transition_cycle_script_v1.json`
- 单铲右区到左区：`testbed/testbed/configs/real_transition_single_cycle_right_to_left_v1.json`
- 单铲左区到右区：`testbed/testbed/configs/real_transition_single_cycle_left_to_right_v1.json`
- 运行脚本：`scripts/run_real_transition_target_release_policy.sh`
- 模型包生成：`scripts/build_real_transition_target_release_bundle.py`
- 静态预检：`scripts/verify_real_transition_target_release_runtime.py`
- 日志检查：`scripts/check_real_transition_target_release_log.sh`

内部 A 表示左区，B 表示右区。示例脚本从右区开始，依次执行左、右、左三铲；它不循环。
示例脚本不用于首次运动。首次运动使用已经准备好的两个单铲脚本，其他顺序仍需单独审核。

## 生成模型包

在训练数据盘可见的开发机执行：

```bash
cd /home/pingfan/Excavator_real_stack
python scripts/build_real_transition_target_release_bundle.py
cd policy_bundles/real_transition_target_release_v2
sha256sum -c SHA256SUMS
```

运行时必须加载 `policy_accepted.ckpt`。训练自动生成的 `policy_best.ckpt` 没有通过最终
冻结门槛，不能替代。

模型包位于 `policy_bundles/`，不进入 git；现场通过已审核的文件复制流程单独同步，并在
Jetson 上再次运行 `sha256sum -c SHA256SUMS`。

## 外置盘自动发现

真机只需要从 GitHub 更新代码，模型包由外置盘提供。外置盘使用下面的相对目录结构：

```text
Excavator_real_stack_runtime/
└── real_transition_target_release_v2_<日期>_<代码提交>/
    └── policy_bundles/
        └── real_transition_target_release_v2/
            ├── policy_accepted.ckpt
            ├── dataset_stats.pkl
            ├── resolved_config.yaml
            ├── accepted_model.json
            ├── runtime_bundle_manifest.json
            ├── SHA256SUMS
            ├── contracts/
            └── evaluation/
```

`run_real_transition_target_release_policy.sh` 会扫描
`/media/*/EXTERNAL_USB*`、`/run/media/*/EXTERNAL_USB*` 和
`/mnt/EXTERNAL_USB*`。它优先使用外置盘上唯一的已验收模型包，并把运行日志写入同一块盘
的 `policy_control_tests/`。新拉取的仓库不需要再复制 `policy_bundles`。

如果找到多个目标释放模型包，脚本会停止并要求人工设置 `BUNDLE_DIR`，不会自行猜测版本。
显式设置的 `BUNDLE_DIR` 和 `LOG_ROOT` 始终优先。没有找到外置盘模型时，开发机才会回退到
仓库内的本地模型包。

## shadow_zero

从端仓库和模型包到位后：

```bash
cd /media/mundane/D/Excavator_real_stack
git switch fs/v2.0.1
git pull --ff-only origin fs/v2.0.1
export MODE=shadow
export CYCLE_SCRIPT=testbed/testbed/configs/real_transition_single_cycle_right_to_left_v1.json
./scripts/run_real_transition_target_release_policy.sh
```

机器在脚本初始区域稳定 0.5 秒后，主端按按钮 4。`shadow_zero` 验证以下内容：

- 明确加载 `policy_accepted.ckpt`；
- 四路相机顺序、qpos 输入和推理正常；
- 第一铲目标被提交，condition 与脚本一致；
- 模型动作有记录，但最终下发动作保持零；
- planner、ready 和停止原因进入逐步日志。

因为下发动作保持零，静态 shadow 不会完成回转位移，也不会自动推进整条目标序列。这是
预期边界，不应伪装成连续脚本验证。

## 受控运动

shadow 日志通过后，准备一条短的、非循环脚本，然后执行：

```bash
cd /media/mundane/D/Excavator_real_stack
export MODE=control
export CYCLE_SCRIPT=testbed/testbed/configs/real_transition_single_cycle_right_to_left_v1.json
export CONFIRM_HARDWARE_MOTION=YES
export CONFIRM_SCRIPT_REVIEWED=YES
./scripts/run_real_transition_target_release_policy.sh
```

第一铲从右区出发，到左区稳定停止后结束。检查日志和机器状态，再把机器保持在左区，使用
`real_transition_single_cycle_left_to_right_v1.json` 单独测试反方向。两个脚本都只有一个
目标且 `loop=false`，不会自动开始第二铲。

按钮 4 的语义：

- 人工模式且初始区域稳定：提交第一目标并进入 ACT；
- ACT 运行中：立即退出到人工；
- 脚本完成、错误区域或超时后：系统锁零；按一次按钮 4 只解除锁零并停留在人工模式。

以下情况不会推进到下一目标：没有完成真实回转位移、目标区域错误、虽在左/右侧但超出
本批训练终点范围、稳定窗不足、传感器时间断裂、周期 60 秒超时或总运行 240 秒超时。

## 日志验收

```bash
cd /media/mundane/D/Excavator_real_stack
export MODE=shadow  # 或 control
./scripts/check_real_transition_target_release_log.sh
```

逐步日志保留模型原始动作、保护后动作、最终下发动作、当前目标、铲次、目标版本、实际
ready 区域、回转位移确认、区域稳定结果和停止原因。

## 证据边界

当前开发机已完成模型包哈希、运行配置静态预检、实际 checkpoint 加载和本地自动测试。
尚未连接现场 Jetson，也没有完成 shadow 或真机动作验证。离线验收和静态预检不能替代
液压响应与真机完整周期结果。
