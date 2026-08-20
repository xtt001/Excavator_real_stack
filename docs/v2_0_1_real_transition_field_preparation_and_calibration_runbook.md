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
3. 核对已由现场确认的 swing home、左右符号和安全范围；
4. 使用现场已确认的 swing 规则自动冻结 `ready_contract.json`，不再采集
   home/A/B 各 10 个固定窗口；
5. 提前冻结首批 sequence/split，保存当天现场证据和 checksum；
6. 判断是否具备进入专家连续录制的条件。

本手册只授权准备、只读检查和人工看护下的 v2 专家录制演练。它不会授权模型动作。任何传感器、版本、方向或安全状态不明确时，当天停在准备阶段。

## 1. 文档和代码边界

- 本文负责 v2.0.1 Real Transition 的 GitHub 交付、准备顺序和规则型 ready 合同。
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
build-ready-contract
verify-run
```

旧版 `init-home-calibration/capture-home-window/build-home-contract` 可以保留在帮助中，
但不再是本轮入口。

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
| 从端 Jetson | bridge、GMSL、gateway、只读状态、session 产物落盘 | policy |
| 主端 | Git/SHA 核对、时间同步、现场记录 | sender、模型控制 |

进入 v2 wrapper 之前的只读检查阶段端口状态：

| 端口 | 期望 |
|---:|---|
| `8766` | C++ bridge 正常监听 |
| `8765` | gateway 正常监听 |
| `8770` | receiver 未监听 |
| `8771` | transition control 未监听 |

进入 v2 前必须停止旧 `policy_remote` receiver，确保 `8770` 释放；随后由 v2 wrapper
启动唯一 expert receiver，并在 `8771` 提供 transition control。

## 5. 准备阶段的安全边界

开始前明确三个人工责任：机手、实验记录人、安全观察人。至少确认：

- 现场急停、熄火、先导关闭和人工接管方式均可用；
- 机器回转半径和铲斗活动范围内无人；
- 通讯或显示中断时机手立即回中并停止；
- ready 确认只读取状态，不启动 policy 动作源；
- 每次 ready 确认前摇杆回中，机器已经停稳且铲斗离土；
- 任何轴方向、IMU、图像或 qpos 解释不一致时停止，不靠修改阈值放行。

从端先清理仓库托管的旧进程，再检查端口：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh status
./scripts/slave_real_stack.sh stop
ss -ltnp | grep -E ':(8765|8766|8770|8771)\b' || true
```

此时四个端口都应为空。只有确认残留进程属于本仓库旧 stack 时，才执行 `./scripts/slave_real_stack.sh stop --force`。未知进程占用端口时先查明来源，不直接杀进程或开始 v2。

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
- 当前道远链 IMU ID `0x121,0x122,0x123,0x124` 全部出现，探针的
  `missing_daoyuan_ids` 为空；
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
test -f "${SESSION_DIR}/ready_contract.json"
test -f "${SESSION_DIR}/preparation_manifest.json"
```

`SESSION_ID` 是首个现场 context 的命名。相机位置、IMU 坐标、home 物理语义或作业区发生实质变化时，关闭当前 context，新建 session 和 contract。后续 session 使用新的、事先记录的 seed，不能复用 `20260817` 后再声称序列独立。

## 9. 已冻结的 swing 规则

2026-08-17 现场已确认以下合同，不再要求移动机器重复机械极限、固定预姿态或
home/A/B 各 10 次窗口：

```text
home swing qpos             = 0.000690 rad
physical left qpos sign     = -1
home tolerance              = 0.05 rad
clean-ready minimum delta   = 0.08 rad
safe swing qpos range       = [-0.3892, +0.4189] rad
swing stable window         = 0.5 s
swing qvel absolute limit   = 0.015 rad/s（全窗每一行均通过）
```

A 固定表示物理左侧，B 固定表示物理右侧。当前 swing qpos 相对 home 的最短角差
小于 `-0.08 rad` 为 clean A，大于 `+0.08 rad` 为 clean B；home tolerance 与 clean
阈值之间是 transition/review 区。安全范围外拒绝 ready。

## 10. 自动生成并复核 ready contract

第 8 节的 `prepare-session` 会同时生成不可变的 `ready_contract.json`，无需输入
`home_calibration_samples.json`：

```bash
test -f "${SESSION_DIR}/ready_contract.json"
"${PYTHON}" -m testbed.cli.real_transition build-ready-contract \
  --output "${SESSION_DIR}/ready_contract.json"
sha256sum "${SESSION_DIR}/ready_contract.json"
```

第二条命令是幂等复核；已有文件内容不同会拒绝覆盖。旧的
`init-home-calibration/capture-home-window/build-home-contract` 仅保留用于读取旧实验，
不属于本轮 v2 现场入口。

## 11. Ready 在线判据

`initial-ready` 和 `target-ready` 均采用同一规则：

```text
当前 swing qpos 自动分类为脚本要求的 clean A/B
+ 最新连续 0.5 s 内每行 abs(swing_qvel) <= 0.015 rad/s
+ 铲斗离土由操作员显式确认
+ 操作员确认当前状态可作为 ready
```

boom、stick、bucket 的 qpos 不设边界；三轴 qvel 的窗口最大值写入
`ready_evidence`，但不阻止 ready。runtime 不再接受人工填写 realized A/B 作为分类
依据；`target-ready` 使用当前 swing qpos 自动计算实际侧，实际侧与 scripted target
不一致时立即 abort。

命令必须显式带两项确认：

```bash
"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 initial-ready \
  --confirm-bucket-clear --confirm-operator-ready

"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 target-ready \
  --confirm-bucket-clear --confirm-operator-ready
```

## 12. HDF5 阈值证据

`episode_111/112/113.hdf5` 的自然停止段用于复核 swing 阈值。规则不是“某一帧低于
0.015”，而是“最新连续 0.5 秒全窗均低于或等于 0.015”。`episode_113` 的长自然
停止段在忽略液压惯性衰减后，`abs(swing_qvel)` p95 约 `0.0066 rad/s`；超阈值点集中
在停止命令后的衰减阶段。因此 `0.015 rad/s + 0.5 s 全窗` 能拒绝尚在滑行的 swing，
同时允许真实静止噪声。

其中 `episode_113.hdf5` 作为本次复现质量最好的主审计样本：共 1164 row、时长
58.316 s；`action`、`observations/qpos`、`observations/qvel`、step/time 数组均为
1164 row、全有限且 step/time 严格递增，`action` 与 `diagnostics/safe_action` 一致。
四路 `video4..video7` 共 4656 帧，无空帧，全部可完整 JPEG 解码为 `216x384x3`；相机
group valid 为 1143/1164，缺失的 21 row 全在启动前段，group skew 的 p50/p95/max
约为 `0.593/1.735/9.353 ms`。控制行间隔 p50/p95/max 约为
`50.073/51.248/76.438 ms`。文件 metadata 标记 `success=1`、`n_steps=1164`，由临时
文件原子落盘，目录中无对应 `.tmp` 或 failed 残留。文件 SHA-256 为：

```text
4e41aebeff36118a78ab488c83b4cf0f8db02289205ed51b83a8eb9186d753b5
```

`episode_112` 的核心数组和四路图像也完整，但相机 group-valid 诊断全程为 false；因此
默认不把它当作本次“完美复现”的基准样本。

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
| operator / reviewer |  |
| control CAN / IMU CAN | `can2` / `can5` |
| camera devices | `video4..video7` |
| ready contract SHA256 |  |
| home / left sign / clean threshold | `0.000690 / -1 / 0.08 rad` |
| USB free space / estimated requirement |  |
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
- [ ] `ready_contract.json` 已由 `prepare-session` 生成并校验；
- [ ] live ready 状态显示自动 A/B、swing 稳定窗和两项人工确认；
- [ ] sequence/split 已在看土面前冻结；
- [ ] 机手、安全观察人、停止条件和 workface 方案已经确认。

任何一项未满足，结论写“准备未通过”，不启动正式 wrapper。

通过后，下一阶段从端唯一入口是：

```bash
EXCAVATOR_TRANSITION_SESSION_DIR="${SESSION_DIR}" \
  ./scripts/run_real_transition_expert_recording.sh
```

这条命令会启动唯一 expert receiver 和 transition control。主端必须使用 transition
配置启动手柄 sender，并由 `tb-real-transition-control` 提交 run/goal/marker。正式录制
前先完成一次不挖土状态机演练。

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
