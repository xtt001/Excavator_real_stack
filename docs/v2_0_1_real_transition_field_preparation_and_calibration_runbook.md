---
type: field-preparation-and-calibration-runbook
project: Excavator Real Stack
version: v2.0.1-real-transition-field-prep-v1
status: code-aligned-not-field-validated
created: 2026-08-17
updated: 2026-08-17
scope: github-delivery-read-only-preflight-home-side-calibration
field_execution_authority: preparation-only
recording_authority: false
branch: fs/v2.0.1
sequence_design: v2_0_1_real_transition_experiment_sequence_design.md
final_scope: v2_0_1_real_transition_final_conclusion.md
general_field_commands: host_slave_start_commands.md
---

# v2.0.1 真机测试与标定准备手册

## 0. 本阶段的交付结果

本阶段完成以下事情：

1. 主端和从端从 GitHub 取得同一份 `fs/v2.0.1`；
2. 验证从端运行环境、底层二进制、外置盘、四路相机、CAN、IMU、qpos/qvel 和时间；
3. 在不启动 receiver、sender 或 policy 的条件下，确认 swing 物理左右符号；
4. 采集 home、A、B 各 10 个独立稳定窗口，并冻结 `home_side_contract.json`；
5. 提前冻结首批 sequence/split，保存当天现场证据和 checksum；
6. 判断是否具备进入专家连续录制的条件。

本手册只授权准备、只读检查和人工看护下的 home/A/B 标定。它不会授权模型动作，也不会自动授权正式录制。任何传感器、版本、方向或安全状态不明确时，当天停在准备阶段。

## 1. 文档和代码边界

- 本文负责 v2.0.1 Real Transition 的 GitHub 交付、准备顺序和 home/A/B 标定。
- 通用主从链路、相机外参、GUI、单轴响应和停止命令以《[主从端启动命令速查](host_slave_start_commands.md)》为准。
- 24 条 run、96 个 cycle 及其平衡规则以《[实验执行序列设计](v2_0_1_real_transition_experiment_sequence_design.md)》为准。
- 任务范围以《[最终结论](v2_0_1_real_transition_final_conclusion.md)》为准。
- 旧的 P0/P1 四-cycle 文档不再用于新 session。

当前代码和本文均未经过 2026-08-17 之后的现场设备验证。现场结果必须保存，不能用本地离线测试代替。

## 2. GitHub 交付合同

### 2.1 唯一取件地址

```text
repository: https://github.com/xtt001/Excavator_real_stack
branch:     fs/v2.0.1
```

主端默认目录：

```text
~/Excavator_real_stack
```

从端默认目录：

```text
/media/mundane/D/Excavator_real_stack
```

两台电脑分别从 GitHub 拉取。不要以其中一台机器的临时副本替代 GitHub，也不要在现场用 `git reset --hard` 清理不明修改。

### 2.2 已有仓库的安全更新

先在各自仓库执行：

```bash
git status --short --branch
git diff --quiet
git diff --cached --quiet
```

后两条任一返回非零，说明存在未提交的 tracked 修改。先保存并确认归属，不继续切分支。未跟踪的现场数据也不要删除或顺手提交。

tracked 工作树干净后执行：

```bash
git fetch --prune origin
git switch fs/v2.0.1
git merge --ff-only origin/fs/v2.0.1

LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/fs/v2.0.1 | awk '{print $1}')"
printf 'local=%s\nremote=%s\n' "${LOCAL_SHA}" "${REMOTE_SHA}"
test -n "${REMOTE_SHA}" && test "${LOCAL_SHA}" = "${REMOTE_SHA}"
```

最后一条必须成功。失败时不要继续上机准备。

如果现有 `origin` 使用了现场不可用的 SSH alias，可以改成公开 HTTPS 读取地址后重试：

```bash
git remote set-url origin https://github.com/xtt001/Excavator_real_stack.git
git fetch --prune origin
```

### 2.3 仓库不存在时

主端可以直接克隆：

```bash
cd ~
git clone --branch fs/v2.0.1 --single-branch \
  https://github.com/xtt001/Excavator_real_stack.git
```

从端先确认 `/media/mundane/D` 是预期数据盘或系统挂载点：

```bash
findmnt /media/mundane/D
```

只有挂载正确且确认仓库确实不存在时，才克隆到约定目录：

```bash
git clone --branch fs/v2.0.1 --single-branch \
  https://github.com/xtt001/Excavator_real_stack.git \
  /media/mundane/D/Excavator_real_stack
```

挂载缺失时先处理挂载，不要在同名空目录里创建另一份仓库。

### 2.4 两端版本必须相同

主端和从端分别执行并记录输出：

```bash
git rev-parse HEAD
git status --porcelain --untracked-files=no
```

通过条件：

- 两端 SHA 完全相同；
- 第二条没有输出；
- 两端 SHA 都等于 GitHub 上 `origin/fs/v2.0.1` 的 SHA。

再确认本轮必需文件属于当前 commit：

```bash
git ls-files --error-unmatch \
  scripts/run_real_transition_expert_recording.sh \
  scripts/slave_real_stack.sh \
  testbed/testbed/cli/real_transition.py \
  testbed/testbed/cli/real_transition_control.py \
  testbed/testbed/tasks/home_side_calibration.py \
  testbed/testbed/tasks/home_side_contract.py \
  testbed/testbed/tasks/real_transition.py \
  testbed/testbed/tasks/real_transition_runtime.py \
  testbed/testbed/configs/teleop_real_transition_v2_0_1.yaml \
  configs/camera_intrinsics/gmsl_h190ta/manifest.json \
  configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json
```

### 2.5 GitHub 会提供和不会提供的内容

GitHub 必须包含源代码、配置、标定采集工具、测试和本文。下面几类内容不会进入公开仓库：

- `.venv`、Conda 环境和本机生成的 C++/CUDA build 目录；
- 现场 raw HDF5、相机标定图片、home/A/B 参考图和 session 产物；
- policy bundle、模型权重和大体积缓存；
- 密码、token、SSH 私钥和现场账号凭据。

因此，两台电脑能从 GitHub 恢复全部软件源，但仍要在各自机器安装环境、编译硬件二进制。现场生成的数据保存在外置盘，并在离场前复制到第二份介质。不要把真机数据或凭据提交到公开 GitHub。

## 3. 从端环境和二进制

### 3.1 Python 环境

从端执行：

```bash
cd /media/mundane/D/Excavator_real_stack

if [[ ! -x .venv/bin/python ]]; then
  ./scripts/setup_env.sh venv
fi

export PYTHON="$PWD/.venv/bin/python"
"${PYTHON}" -m pip install --no-build-isolation --no-deps -e ./testbed
"${PYTHON}" -m testbed.cli.real_transition --help
./scripts/check_target_prereqs.sh
```

`tb-real-transition --help` 必须列出以下子命令：

```text
prepare-session
init-home-calibration
capture-home-window
build-home-contract
verify-run
```

### 3.2 生成目录不在 Git 中，需在从端编译

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate

cmake -S bridge -B bridge/build \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
cmake --build bridge/build --target excavator_real_bridge -j2

cmake -S tools/gmsl_realtime_capture \
  -B tools/gmsl_realtime_capture/build
cmake --build tools/gmsl_realtime_capture/build -j2

test -x bridge/build/excavator_real_bridge
test -x tools/gmsl_realtime_capture/build/gmsl_realtime_preprocess_probe
```

第二个 binary 只会在 Jetson CUDA 和 multimedia 依赖完整时生成。缺失时停止相机链路准备，不要切换成未经本轮批准的相机实现。

## 4. 现场角色和端口

| 设备 | 本阶段职责 | 本阶段不启动 |
|---|---|---|
| 从端 Jetson | bridge、GMSL、gateway、只读状态、标定产物落盘 | receiver、policy |
| 主端 | Git/SHA 核对、时间同步、现场记录 | sender、模型控制 |

准备和 home/A/B 采集期间的端口状态：

| 端口 | 期望 |
|---:|---|
| `8766` | C++ bridge 正常监听 |
| `8765` | gateway 正常监听 |
| `8770` | receiver 未监听 |
| `8771` | transition control 未监听 |

`capture-home-window` 只读取从端本机 `127.0.0.1:8765`。它会检查 `8770` 未监听，缺少四路相机、相机时间戳不前进、黑帧或稳定窗超限时直接拒绝样本。

## 5. 准备阶段的安全边界

开始前明确三个人工责任：机手、实验记录人、安全观察人。至少确认：

- 现场急停、熄火、先导关闭和人工接管方式均可用；
- 机器回转半径和铲斗活动范围内无人；
- 通讯或显示中断时机手立即回中并停止；
- 标定只用人工驾驶改变姿态，软件侧没有 sender、receiver 或 policy 动作源；
- 每个稳定窗口开始前摇杆回中，机器已经停稳；
- 任何轴方向、IMU、图像或 qpos 解释不一致时停止，不靠修改阈值放行。

从端先清理仓库托管的旧进程，再检查端口：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh status
./scripts/slave_real_stack.sh stop
ss -ltnp | grep -E ':(8765|8766|8770|8771)\b' || true
```

此时四个端口都应为空。只有确认残留进程属于本仓库旧 stack 时，才执行 `./scripts/slave_real_stack.sh stop --force`。未知进程占用端口时先查明来源，不直接杀进程或开始标定。

## 6. 外置盘、时间和静态检查

### 6.1 外置盘

从端执行：

```bash
findmnt /media/mundane/EXTERNAL_USB
df -h /media/mundane/EXTERNAL_USB
test -w /media/mundane/EXTERNAL_USB
mkdir -p /media/mundane/EXTERNAL_USB/real_transition_raw_v2
```

不预设一个没有现场文件大小证据的固定 GB 数。先用短时传感器冒烟得到实际写入速率，再按下面公式留盘：

```text
所需空间 >= 单条最长 5-cycle run 的实测大小 × 30
```

30 包含 24 条正式 run 和失败/重录余量。计算后再保留至少 20% 的文件系统空闲空间。空间不足时减少当天计划或换盘，不改写已冻结的原始文件。

### 6.2 主从时间

两台现场电脑接入直连网后，在主端执行：

```bash
cd ~/Excavator_real_stack
./scripts/sync_slave_time_from_host.sh \
  --ssh-host slave-jetson \
  --ssh-user mundane
```

脚本当前通过线是绝对时间差不超过 5 秒。命令失败时不要用文件修改时间判断 run 先后，先修复主从时间。

### 6.3 设备静态检查

从端执行：

```bash
ls -l /dev/video4 /dev/video5 /dev/video6 /dev/video7
ip -details -statistics link show can2
ip -details -statistics link show can5
```

当前合同为：控制 CAN `can2`，IMU CAN `can5`，相机 `video4..video7`。当天若设备号或接口发生变化，先确认接线和配置 owner，再建立新的 context；不要只在命令行临时替换后继续沿用旧 context。

## 7. 只读传感器链路

从端终端 A 启动无 receiver 链路：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force --no-receiver
```

从端终端 B 检查：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh status
./scripts/imu_can_probe.py \
  --interface can5 \
  --duration-s 3 \
  --require-four
```

通过条件：

- `camera_mode=gmsl`，`imu_mode=hardware`；
- bridge、GMSL、gateway 为 running；
- `8765/8766` 有监听，`8770/8771` 空闲；
- 四个 GMSL SHM 均存在；
- IMU raw addr `0,1,2,3` 全部出现；
- `can2/can5` 没有 BUS-OFF。

随后记录至少 60 秒 qpos/qvel：

```bash
./scripts/log_imu_qvel_quality.py \
  --rate-hz 50 \
  --duration-s 60 \
  --print-every-s 1 \
  --verbose-imu
```

检查 qpos、qvel 和 raw IMU 连续，没有 NaN、跳变、placeholder 或长时间 stale。不要在真机只读检查中运行 `scripts/smoke_bridge_protocol.py`；该脚本包含非零 `send_action` 协议测试。

相机内参和 preprocess manifest 已随 Git 提供。每天先核对镜头、序列号映射、画面方向、遮挡和安装松动。相机安装未改变时无需每天重做内参。`video4/video5` 安装发生变化，或后续任务需要几何投影时，再按通用主从文档的 eye 外参流程重标；现有通过候选为有效 pair 至少 12 对、RMS 小于 2 px，并需人工复查角点覆盖。

## 8. 冻结首批 session

首批正式 session 的 sequence seed 固定为 `20260817`。这个数在查看当天土面和目标顺序之前确定。不要因为序列看起来不方便而重新抽 seed。

从端执行：

```bash
cd /media/mundane/D/Excavator_real_stack
export PYTHON="$PWD/.venv/bin/python"
export SESSION_ID="rt_$(date +%Y%m%d)_ctx01"
export SESSION_ROOT="/media/mundane/EXTERNAL_USB/real_transition_raw_v2"

"${PYTHON}" -m testbed.cli.real_transition prepare-session \
  --output-root "${SESSION_ROOT}" \
  --session-id "${SESSION_ID}" \
  --seed 20260817

export SESSION_DIR="${SESSION_ROOT}/session_${SESSION_ID}"
test -f "${SESSION_DIR}/sequence_manifest.json"
test -f "${SESSION_DIR}/split_manifest.json"
test -f "${SESSION_DIR}/preparation_manifest.json"
```

`SESSION_ID` 和 `ctx01` 只是首个现场 context 的命名。相机位置、IMU 坐标、home 物理语义或作业区发生实质变化时，关闭当前 context，新建 session 和 contract。后续 session 使用新的、事先记录的 seed，不能复用 `20260817` 后再声称序列独立。

## 9. 确认 swing 物理左右符号

此步骤仍保持 `--no-receiver`，主端不启动 sender。

1. 机手把机器置于可安全观察的 home 附近，停稳并记录 `q_home[0]`；
2. 机手用原车人工控制向物理左侧移动一小段，停稳并记录 `q_left[0]`；
3. 视觉确认运动方向确实是物理左侧；
4. 计算最短角差：

```text
delta_left = atan2(sin(q_left[0] - q_home[0]), cos(q_left[0] - q_home[0]))
delta_left > 0  => physical_left_qpos_sign = +1
delta_left < 0  => physical_left_qpos_sign = -1
```

符号确认的观测差建议达到 `abs(delta_left) >= 0.05 rad`，同时保持现场认为安全的位移。差值太小、跨角分支不清楚或视觉与 qpos 不一致时，符号未通过。不要根据摇杆 action 或历史配置猜符号。

把最终符号、两次稳定 qpos、时间、机手和复核人写进当天现场记录。

## 10. 初始化 home/A/B 标定输入

以下示例假设确认结果为 `+1`。若现场结果为 `-1`，只改 `LEFT_SIGN`，不要改代码默认值。

```bash
cd /media/mundane/D/Excavator_real_stack
export PYTHON="$PWD/.venv/bin/python"
: "${SESSION_DIR:?先按第 8 节设置 SESSION_DIR}"

export LEFT_SIGN=1
export CONTEXT_VERSION="${SESSION_ID}"
export OPERATOR_ID="field_engineer_01"

"${PYTHON}" -m testbed.cli.real_transition init-home-calibration \
  --output "${SESSION_DIR}/home_calibration_samples.json" \
  --context-version "${CONTEXT_VERSION}" \
  --resolved-by "${OPERATOR_ID}" \
  --physical-left-qpos-sign "${LEFT_SIGN}" \
  --source-config testbed/testbed/configs/teleop_real_v1.yaml \
  --source-value-path teleop.recording.go_home.home_pose_rad \
  --expected-cameras video4,video5,video6,video7
```

命令会把 home 配置原文复制到 session 内，再创建可追加的 calibration JSON。这样离开当前仓库绝对路径后仍能复核来源。配置里的 `home_pose_rad` 是命令参考；实际分类中线由 10 次独立回中解析。

## 11. 采集 30 个独立稳定窗口

### 11.1 每个窗口的自动门槛

`capture-home-window` 每次采 0.5 秒、名义 20 Hz，并检查：

| 项目 | 门槛 |
|---|---:|
| software action source | receiver `8770` 未监听，并由实验员确认无 sender/policy |
| 四路相机 | `video4..video7` 每路存在、时间戳前进、图像均值 `>5` |
| 稳定时间 | `>=0.5 s` |
| qvel 绝对值 | `<= [0.015, 0.015, 0.020, 0.020] rad/s` |
| qpos 峰峰值 | 每轴 `<=0.005 rad` |
| 视觉 | 实验员确认当前姿态和作业面可作为 ready |

通过后才会把 qpos/qvel、status、四路 JPEG、时间戳和 provenance 追加到 calibration 文件。失败窗口不会进入 accepted 样本。

### 11.2 单个命令

机器停稳并完成视觉检查后执行：

```bash
"${PYTHON}" -m testbed.cli.real_transition capture-home-window \
  --calibration "${SESSION_DIR}/home_calibration_samples.json" \
  --side home \
  --reference-id home_01 \
  --host 127.0.0.1 \
  --port 8765 \
  --receiver-port 8770 \
  --duration-s 0.5 \
  --rate-hz 20 \
  --confirm-visual \
  --confirm-no-software-action-source
```

后续只改 `--side` 和 `--reference-id`，例如 `home_02..home_10`、`A_01..A_10`、`B_01..B_10`。

不要写 shell 循环连续采 10 次。独立性要求如下：

- home：每次先离开中线，再由机手重新回中、停稳、确认；
- A/B：每次重新选择该侧自然可挖的 ready，覆盖当天实际会使用的位置变化；
- 同一次停稳后的连续 10 个窗口不算 10 次独立参考；
- A、B 是区域，不要求对准固定角度；
- 每次命令成功后查看 JSON 输出中的计数和 qvel/qpos 峰峰值，再进行下一次。

四路参考图保存在：

```text
${SESSION_DIR}/calibration_visuals/<reference_id>/video4.jpg
${SESSION_DIR}/calibration_visuals/<reference_id>/video5.jpg
${SESSION_DIR}/calibration_visuals/<reference_id>/video6.jpg
${SESSION_DIR}/calibration_visuals/<reference_id>/video7.jpg
```

## 12. 生成和复核 home-side contract

30 个 accepted 窗口完成后执行：

```bash
"${PYTHON}" -m testbed.cli.real_transition build-home-contract \
  --calibration "${SESSION_DIR}/home_calibration_samples.json" \
  --output "${SESSION_DIR}/home_side_contract.json"

sha256sum \
  "${SESSION_DIR}/home_calibration_samples.json" \
  "${SESSION_DIR}/home_side_contract.json"
```

生成器按当天数据解析：

```text
classification_deadband_rad = max(
  0.05,
  ceil_to_0.01(max(abs(center_repeat_error)) + 0.01)
)

clean_endpoint_min_abs_side_coordinate_rad =
  classification_deadband_rad
  + max(0.03, p95(abs(center_repeat_error)))
```

历史下限为 deadband `0.05 rad`、clean endpoint `0.08 rad`。当天回中波动更大时数值只会增大。生成失败时不要手工缩小 deadband 或删除不方便的方向证据来凑通过。

复核以下内容：

```bash
"${PYTHON}" - "${SESSION_DIR}/home_side_contract.json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding="utf-8"))
print("context:", d["context_version"])
print("home:", json.dumps(d["home_reference"], indent=2))
print("counts:", d["parameter_resolution"]["observed_statistics"]["accepted_window_counts"])
print("sides:", json.dumps(d["parameter_resolution"]["observed_statistics"]["sides"], indent=2))
print("contract_sha256:", d["contract_sha256"])
PY
```

通过条件：

- home/A/B 均至少 10 个 accepted 窗口；
- A 全部位于物理左侧 clean 支持，B 全部位于物理右侧 clean 支持；
- 两侧最近支持点都越过 clean endpoint；
- 四路参考图与 qpos 分类一致；
- 没有放宽 ready 门槛的未审批 override；
- contract 和源配置 snapshot 均有 SHA256。

## 13. 准备证据封存

在正式录制前，从端执行：

```bash
(
  cd "${SESSION_DIR}"
  find . -type f ! -name preparation_files.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "${SESSION_DIR}/preparation_files.sha256"

(
  cd "${SESSION_DIR}"
  sha256sum -c preparation_files.sha256
)
```

然后把整个 session 准备目录复制到第二份介质。副本用于灾备，不作为另一个可写主目录。后续正式 raw run 仍只写原 session。

当天现场记录至少填写：

| 字段 | 值 |
|---|---|
| main SHA |  |
| slave SHA |  |
| GitHub remote SHA |  |
| session id / seed |  |
| context version |  |
| operator / reviewer |  |
| control CAN / IMU CAN | `can2` / `can5` |
| camera devices | `video4..video7` |
| physical-left qpos sign and evidence |  |
| home/A/B accepted counts |  |
| deadband / clean threshold |  |
| USB free space / estimated requirement |  |
| contract SHA256 |  |
| preparation checksum result |  |

## 14. 是否进入正式录制

全部满足才可以进入下一阶段：

- [ ] 两端 checkout 与 GitHub remote SHA 一致，tracked 工作树干净；
- [ ] Python 环境和两类本机 binary 可用；
- [ ] 外置盘挂载、可写、空间估计通过；
- [ ] 主从时间差不超过 5 秒；
- [ ] `can2/can5` 正常，四个 IMU 地址齐全；
- [ ] 四路相机设备、SHM、画面、时间戳和安装状态正常；
- [ ] qpos/qvel 连续且坐标解释一致；
- [ ] `physical_left_qpos_sign` 有视觉和 qpos 证据；
- [ ] home/A/B 各 10 个独立窗口通过；
- [ ] `home_side_contract.json` 生成、复核并封存；
- [ ] sequence/split 已在看土面前冻结；
- [ ] 机手、安全观察人、停止条件和 workface 方案已经确认。

任何一项未满足，结论写“准备未通过”，不启动正式 wrapper。

通过后，下一阶段从端唯一入口是：

```bash
EXCAVATOR_TRANSITION_SESSION_DIR="${SESSION_DIR}" \
  ./scripts/run_real_transition_expert_recording.sh
```

这条命令会启动 receiver 和 transition control，因此不属于本文前面的只读标定阶段。主端必须使用 transition 配置启动手柄 sender，并由 `tb-real-transition-control` 提交 run/goal/marker。正式操作顺序应在首条 run 前单独做一次全员口头演练。

## 15. 需要动作的测试单独审批

下面几项会发送真实控制命令，不能混入只读准备：

- 主端 remote sender；
- `calibrate_axis_response.py`；
- go-home；
- policy shadow 之外的 control；
- 正式 transition receiver。

只有液压、手柄映射或动作方向确实发生变化时，才按通用主从文档重新做单轴响应标定。每次只测一个轴，现场设置动作上限、停止位移、观察人和人工急停。标定结果进入新的 resolved config；不要直接覆盖已经封存的 session 证据。

## 16. 停止与异常保留

只读准备结束时，在运行 `slave_real_stack.sh run --force --no-receiver` 的终端按一次 `Ctrl+C`。然后执行：

```bash
./scripts/slave_real_stack.sh status
sync /media/mundane/EXTERNAL_USB
```

如果正式录制已经开始，停止顺序不同：先停主端 sender，让动作归零；等待从端显示 `SAVING` 完成；再停日志和从端 stack。中断、失败窗口、错误输出和现场修改记录都保留，不用成功样本覆盖。
