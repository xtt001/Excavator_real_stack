# v2.0.1 主从端现场启动与命令手册

本文是 `fs/v2.0.1` 专家连续录制的唯一现场入口。旧 teleop、`policy_remote` 和
N5/E52 流程仍参考《[主从端启动命令速查](host_slave_start_commands.md)》，不要与本文
混用。

## 1. 固定合同

| 项目 | 固定值 |
|---|---|
| 主端仓库 | `~/Excavator_real_stack` |
| 从端 SSH | `slave-jetson`，即 `mundane@192.168.100.1` |
| 从端仓库 | `/media/mundane/D/Excavator_real_stack` |
| 分支 | `fs/v2.0.1`；两端 `git rev-parse HEAD` 必须一致 |
| 当前新 session | `/media/mundane/EXTERNAL_USB/real_transition_raw_v2/session_rt_20260822_ctx04` |
| 控制 CAN / IMU CAN | `can2` 250 kbit/s / `can5` 1 Mbit/s |
| 四路相机 | `/dev/video4..7` |
| bridge / gateway | `127.0.0.1:8766` / `127.0.0.1:8765` |
| sender → receiver | `192.168.100.1:8770` |
| v2 恢复/诊断 API | `192.168.100.1:8771`；正常操作不用 |
| sender → 主端 GUI | `127.0.0.1:8781/UDP` |

本流程不加载 policy，不启动 `policy_remote`。左手柄物理按钮 2 只在整个 session 开始前
按一次，用于 `ARM SESSION`；它不再产生 cycle marker。ARM 后的正常状态流为：

```text
等待下一条 initial side 稳定 -> 自动 start-run + initial-ready
initial-ready                 -> 自动提交冻结 goal
goal_committed                -> 正常完成挖掘/回转/卸料
swing 相对本 cycle 起点持续移动 >=0.08 rad -> 自动记录 cycle_excursion_observed
回到 TARGET 且 swing 稳定      -> 自动 target-ready / 下一 goal / 保存
```

操作员在 run/cycle 内不做任何专门标注按键，也不离开摇杆点鼠标。GUI 只显示 ARM、
TARGET、自动检测阶段、blocker、相机同步和保存结果，不提交任何事件。`dump-end` 不再是
在线状态机门槛；materializer 只保留 return proxy，不把代理点冒充人工确认的 dump。

左手柄物理按钮 3 仍是主动 go-home；它不是正常 run 的必需步骤。左手柄物理按钮 4
用于取消当前 run，不切换 policy。机器状态仍使用现场既有按钮。

ready 合同固定为：

```text
home swing qpos           = 0.000690 rad
物理左侧 qpos 符号       = -1
home tolerance            = 0.05 rad
clean-ready 最小偏移      = 0.08 rad
安全 swing 范围           = [-0.3892, +0.4189] rad
swing 稳定窗              = 连续 0.5 s
swing qvel 上限           = 全窗每行 abs(qvel) <= 0.015 rad/s
```

`A` 是物理左侧，`B` 是物理右侧。boom、stick、bucket 可自然预摆，三轴 qpos 不设
ready 边界，qvel 只记录、不阻止 ready。一次 session ARM 表示操作员授权本 session 的
自动边界；系统无法直接测量铲斗是否入土，因此 `bucket_clear` 改为录后画面/QC 项，不再
伪装成逐 cycle 的人工确认。

## 2. 每次启动前检查

### 2.1 主端

```bash
cd ~/Excavator_real_stack

ip route get 192.168.100.1
ping -c 3 192.168.100.1

git branch --show-current
git rev-parse HEAD
git status --porcelain --untracked-files=no

ssh slave-jetson \
  'cd /media/mundane/D/Excavator_real_stack && \
   git branch --show-current && git rev-parse HEAD && \
   git status --porcelain --untracked-files=no'

./scripts/sync_slave_time_from_host.sh \
  --ssh-host slave-jetson \
  --ssh-user mundane
```

两端必须在同一 commit，tracked 工作树无输出。日志、build 和备份等 untracked 目录不
因此删除。

### 2.2 从端

```bash
ssh slave-jetson
cd /media/mundane/D/Excavator_real_stack

findmnt /media/mundane/EXTERNAL_USB
df -h /media/mundane/EXTERNAL_USB
ls -l /dev/video4 /dev/video5 /dev/video6 /dev/video7
ip -details -statistics link show can2
ip -details -statistics link show can5

./scripts/slave_real_stack.sh status
ss -ltnp | grep -E ':(8765|8766|8770|8771)\b' || true
```

若 `8770` 被旧 `policy_remote` 或 v1 receiver 占用，先在原终端 `Ctrl+C`。确认进程属于
本仓库托管栈后也可执行：

```bash
./scripts/slave_real_stack.sh stop
```

如果 `/dev/video4..7` 不存在，先执行当天首次 GMSL bring-up；设备均存在时无需重复：

```bash
GMSL_VIDEO_DEVICES="4 5 6 7" ./scripts/bring_up_gmsl_cameras.sh
```

## 3. 固定启动顺序

以下持续进程各占一个终端。运动前确认现场急停、熄火、先导关闭和人工接管均可用，
机器工作范围内无人。

### 3.1 从端终端 A：唯一 v2 主链路

```bash
cd /media/mundane/D/Excavator_real_stack

export EXCAVATOR_TRANSITION_SESSION_DIR="/media/mundane/EXTERNAL_USB/real_transition_raw_v2/session_rt_20260822_ctx04"
./scripts/run_real_transition_expert_recording.sh
```

wrapper 会依次完成：

1. 挂载并检查外置盘和 session 的 immutable artifacts；
2. 配置 `can2/can5`，设为 Jetson MAXN 并锁 CPU/GPU/EMC；
3. **每次 capture 启动前**重新给 `/dev/video4..7` 写入
   `sensor_mode=2,trig_mode=2,trig_pin=0x00020007`，解决 Jetson/GMSL 重启后 video7
   脱离外触发的问题；
4. 启动 bridge、四路 GMSL SHM 和 gateway；
5. 在 receiver 启动前采样 3 s 四相机 group metadata。只有有效率 `>=0.98`、最大
   group skew `<=5 ms` 且 distinct group `>=30` 才启动 receiver；
6. 启动 expert receiver 和 `8771` 恢复/诊断 API。

现场 SG8A 驱动会持续给 `video4/video5` 设置 `V4L2_BUF_FLAG_ERROR`，但既有动态画面、
sequence/timestamp、CUDA preprocess 和 SHM 证据均正常。该原始 flag 必须继续写入 QC，
但不能单独把同步组判死；正式门禁仍要求四路同 group、`group_valid=1`、时间戳推进、
group 数足够且 skew 不超过 5 ms。若同时出现 group 无效、时间戳停滞、缺帧或画面异常，
仍按相机故障处理。

如果预检失败，`8770` 不会监听，因此不可能进入录制。不要用 `--no-camera` 绕过正式
录制门禁；先按第 8 节检查触发配置、日志和四路 group metadata。

### 3.2 从端终端 B：GUI 的 video4 预览

等待四路 GMSL SHM 建立后执行：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/start_gmsl_eye_stream.sh
```

这只发布 GUI 预览。四路训练图像仍由从端 gateway/HDF5 链路保存。

### 3.3 主端终端 C：双摇杆 sender

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"

"${PYTHON}" -m testbed.cli.teleop_remote \
  --config testbed/testbed/configs/teleop_real_transition_v2_0_1.yaml \
  --host 192.168.100.1 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --confirm-remote-control
```

轴绑定固定为：左摇杆 Y = swing、左摇杆 X = stick、右摇杆 Y = boom、右摇杆 X =
bucket；动作数组为 `[swing, boom, stick, bucket]`。sender 必须识别两只摇杆。物理
按钮 2 的日志应显示一次 `mark=1`；它只 ARM 整个 session。之后重复按不会创建标注。

### 3.4 主端终端 D：只读 GUI

```bash
cd ~/Excavator_real_stack

EXCAVATOR_TELEOP_CONFIG="$PWD/testbed/testbed/configs/teleop_real_transition_v2_0_1.yaml" \
  ./scripts/start_host_dashboard.sh --always-on-top
```

GUI 不连接控制端口、不发送 action，也不向 `8771` 写事件。顶部 AGE 是数据新鲜度，
不是控制执行时延；10 Hz 状态镜像出现约 `0–100 ms` age 正常。
v2 面板固定显示中文任务字段 `当前起始点=A/B`、`下次目标位置=A/B`、
`当前位置=A/B/home/transition`，run 内同时显示自动检测状态、
`excursion=YES/NO`、swing 稳定窗 `当前/0.50s`、自动等待原因和最近一次自动事件。
`已录制条数` 以当前 v2 session 已封存的 run 数为准，不使用旧版顶层 episode 扫描；
`cycle=X/N` 中 `X` 表示当前 cycle（已完成数加一），等待下一条 run 时从 `1/N` 开始。
四路 IMU 卡中央显示当前四轴 qpos；绿/黄/红框和下方小字表示 IMU 在线、姿态有效性、
数据年龄及非零丢包，不再用大号 `ONLINE` 占据主要空间。
录制中顶部状态和录制卡显示橙色 `● REC 正在录制`，预览画面出现橙框，并持续提示
“左手柄物理按钮 4：取消并重录本 run”。出现 `saving_run` 时禁止关闭、重启或拔盘；
正常 run 内 GUI 不要求点击或 MARK。红色只用于拒绝、故障和危险警告；操作提示使用
深色底与高对比白字。

## 4. 点火前放行条件

GUI 必须同时显示：

- `SENDER / RECEIVER / BRIDGE / VIDEO4` 健康，`CONTROL ACK` 正常；
- 四路 IMU 全部 ONLINE，无 stale/NaN；
- v2 面板显示新 session 和下一条 run；
- `录制前相机同步=PASS`，没有 camera blocker；
- receiver 为 `armed`，没有 policy/model control；
- 两只摇杆回中，qpos/qvel、画面和物理方向一致；
- 外置盘已挂载、可写且空间足够。

机器状态沿用当前现场顺序：

```text
左手柄按钮 5：remote mode
左手柄按钮 1：ignition
左手柄按钮 6：pilot
```

三项打开后 GUI 期望：

```text
status11 = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
```

状态按钮出现在 sender 日志不代表机器必然执行；还要核对 receiver/bridge ACK、
`status11`，必要时查看 CAN 状态帧。

## 5. 正式录制：session 只 ARM 一次，run 内零标注按键

1. 确认整场 session 可以开始后，只按一次左手柄物理按钮 2。GUI 显示
   `SESSION ARMED=YES`；此后不要再为标注按该键。
2. 将 swing 放到 GUI 要求的下一条 initial clean A/B，大小臂和铲斗可自然预摆。摇杆
   回中且 swing 稳定窗通过后，程序自动选择 frozen run、写入 `wf_NNN/fresh_strip`、
   启动 HDF5，并在第一条精确 row 自动写 initial-ready 与 goal。
3. 看清 GUI 的 TARGET 后自然完成挖掘、回转和实际卸料。过程中不按 MARK；程序只观察
   数据，不生成任何机器动作。
4. 当 swing 相对本 cycle 的 goal 起点连续 3 个采样移动至少 `0.08 rad` 时，程序记录
   `cycle_excursion_observed`，证明本 cycle 不是停在原地误完成。这里是相对行程检测，
   不是要求或允许机器越过 `[-0.3892,+0.4189] rad` 的安全范围。
5. 自然回到 TARGET 侧。最近 0.5 s 全窗满足 clean side、采样连续且
   `abs(swing_qvel)<=0.015 rad/s` 后，程序自动写 target-ready。
6. 若还有 cycle，下一 goal 在同一 row 自动提交；继续正常操作。完成 frozen 3/4/5
   cycle 后自动停止、保存和封存，不需要 dump/target 点击或按键。
7. 两条 run 之间，为避免相同 initial side 被立即误启动，程序要求观察到一次自然的
   操作员调整，再等下一 initial 稳定后自动开始。等 GUI 显示保存完成后继续即可。

需要放弃当前 run 时，按一次左手柄物理按钮 4。若 HDF5 尚未开始，空 run 包会直接
回收；若已经开始录制，部分数据会封存到 session 的 `cancelled_runs/` 供诊断，但不计入
已完成条数，也不消耗 frozen sequence，下一次仍重录同一个 run。按下后等待 GUI 退出
红色 `REC` 状态，不要连续重复按键。

如果自动流程没有推进，只看 GUI 的 `automatic_wait_reason`：通常是等待 initial side、
等待本 cycle 形成有效 swing 行程、等待 TARGET 稳定或等待两条 run 之间的自然调整。不要为此
反复按键，也不要求 run 中 go-home。软件无法直接证明铲斗离土；该项在 materializer/QC
中保留为录后画面审计，而不是打断操作员的在线确认。

## 6. 每条 run 的验证和 cycle 生成

### 6.1 验证原始 run package

```bash
cd /media/mundane/D/Excavator_real_stack
export PYTHON="$PWD/.venv/bin/python"
export SESSION_DIR="/media/mundane/EXTERNAL_USB/real_transition_raw_v2/session_rt_20260822_ctx04"

find "${SESSION_DIR}" -name run_manifest.json -printf '%T@ %h\n' \
  | sort -n | tail -n 1

"${PYTHON}" -m testbed.cli.real_transition verify-run \
  "${SESSION_DIR}/block_b01/run_b01_r01"
```

把最后路径替换成 `find` 输出。只有 `status: PASS` 且 run status 为 `complete` 才作为
成功 run；aborted 数据仍保留。

### 6.2 raw run 直接生成 20 Hz cycle

单条验证或试录：

```bash
"${PYTHON}" -m testbed.cli.real_transition materialize-run \
  "${SESSION_DIR}/block_b01/run_b01_r01" \
  --output-dir "/media/mundane/EXTERNAL_USB/real_transition_cycle_v1/run_b01_r01_v1"
```

整个 session：

```bash
"${PYTHON}" -m testbed.cli.real_transition materialize-session \
  "${SESSION_DIR}" \
  --output-dir "/media/mundane/EXTERNAL_USB/real_transition_cycle_v1/session_rt_20260822_ctx04_v2"
```

输出目录必须不存在，工具拒绝覆盖。生成内容包括：

```text
annotations/cycle_annotations_v2.jsonl
episodes/episode_<N>.hdf5
cycle_manifest.jsonl
split_manifest.json
train_ready_manifest.json
resolved_materializer_config.json
SHA256SUMS.txt
```

每个 ready-to-ready cycle 都带四路 JPEG、qpos/qvel/action、20 Hz 时间戳、
`real_transition_condition_v1=[target_side_code,1]`、20-step `valid_mask` 和 source row
provenance。50 ms 网格选择第一条不早于目标时刻的 source row；action 标签固定偏移
`-20 ms`。

materializer 会按 recorded group metadata、时间 gap、teleop action source、目标一致性和
`goal_lead_ms` 自动分为 `clean/review/excluded`。只有 clean episode 进入
`train_ready_manifest.json`；原始 run 永不修改。

## 7. 正常停止与异常处理

正常结束：

1. 完成当前 run，等待 GUI 显示保存完成；
2. 主端 sender 按一次 `Ctrl+C`，确认 action 回零；
3. 按现场安全顺序关闭 pilot、ignition、remote；
4. 关闭 GUI 和 eye stream；
5. 从端 v2 wrapper 按一次 `Ctrl+C`，等待 receiver 保存/sync 后其余服务退出；
6. 检查端口均释放。

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh status
ss -ltnp | grep -E ':(8765|8766|8770|8771)\b' || true
sync /media/mundane/EXTERNAL_USB
```

异常或危险时先使用驾驶室急停、熄火、先导关闭等硬件措施，不依赖 GUI。GUI 已是只读，
软件侧需要记录异常时使用第 9.3 节的 `safety-stop`/`abort` CLI。不要在 `saving` 阶段
直接强杀 receiver；失败 run、事件和日志全部保留，不删除后复用 run ID。

## 8. GMSL/video7 同步诊断

wrapper 会自动执行以下合同。仅诊断时手工查看：

```bash
for dev in 4 5 6 7; do
  v4l2-ctl -d "/dev/video${dev}" --get-fmt-video
  v4l2-ctl -d "/dev/video${dev}" \
    --get-ctrl bypass_mode,sensor_mode,trig_mode,trig_pin
done
```

期望四路均为 `1920x1536 UYVY`、`sensor_mode=2`、`trig_mode=2`、
`trig_pin=0x00020007`。gateway 已启动时可独立复核：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/check_gmsl_sync.py \
  --host 127.0.0.1 --port 8765 \
  --duration-s 3 --rate-hz 20 \
  --min-valid-fraction 0.98 \
  --max-skew-ms 5 \
  --min-distinct-groups 30
```

输出必须为 `PASS`。失败时查看：

```bash
./scripts/slave_real_stack.sh tail gmsl
./scripts/slave_real_stack.sh tail gateway
ls -l /dev/shm/excavator_gmsl_video{4,5,6,7}
```

不要只重启 receiver；video7 脱同步时应正常停止整套 stack，再由 wrapper 重新应用四路
触发配置并通过预检。

若机器或链路在 `saving` 阶段故障，run 目录可能只剩 `.raw.hdf5.tmp.<pid>` 和
`task_events.jsonl`。不要改写 `session_manifest.json`，也不要把缺少 action/timestamps 的
临时 HDF5 改名冒充成功数据。完整保留并把该 session 改名为 `interrupted_*`，使用相同
seed 创建新 session 后从未封存的 run 重新录制。receiver 报
`refusing to overwrite immutable artifact: .../session_manifest.json` 时，先按此规则检查，
不能反复强启同一损坏 session。

如果 session 根目录尚未出现任何 `block_*`/`run_*`、HDF5 或事件数据，receiver 会自动
修复空文件，或更新因代码/配置变化而过期的 `resolved_record_config.yaml` 和
`session_manifest.json`，不需要递增新的 `ctx`。一旦出现任何 run 目录，运行期 artifact
继续严格锁定；此时的冲突必须先检查并保留数据。所有 immutable artifact 采用“先写临时
文件、fsync、再原子发布”，掉电不会再把 0 字节临时结果暴露为正式 manifest。
已有 run 后若仅代码 commit 更新而数据配置与合同未变，receiver 复用原 session manifest；
每条新 run 仍在自身 provenance 中记录当前 commit。配置或合同变化仍会拒绝混入同一 session。

## 9. 命令附录

### 9.1 从端 stack

```bash
./scripts/slave_real_stack.sh --help
./scripts/slave_real_stack.sh status
./scripts/slave_real_stack.sh logs
./scripts/slave_real_stack.sh tail
./scripts/slave_real_stack.sh tail receiver
./scripts/slave_real_stack.sh tail bridge
./scripts/slave_real_stack.sh tail gmsl
./scripts/slave_real_stack.sh tail gateway
./scripts/slave_real_stack.sh stop
./scripts/slave_real_stack.sh mount-usb
```

底层还支持 `start`、`run`、`restart` 以及 `--no-camera`、`--no-imu`、`--no-receiver`、
`--skip-usb`、`--skip-can` 等诊断覆盖；正式 v2 录制不得用降级参数。`--policy-remote`
属于旧流程。

### 9.2 主端 sender / GUI

```bash
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m testbed.cli.teleop_remote --help

# sender 无终端 monitor，仅诊断显示负载
"${PYTHON}" -m testbed.cli.teleop_remote \
  --config testbed/testbed/configs/teleop_real_transition_v2_0_1.yaml \
  --host 192.168.100.1 --port 8770 --input joystick --rate-hz 50 \
  --no-monitor --confirm-remote-control

# GUI 参数
PYTHONPATH="$PWD/testbed" /usr/bin/python3 \
  -m testbed.cli.host_dashboard --help
```

### 9.3 v2 状态和异常恢复 CLI

正常 run 不使用这些命令；它们不发送摇杆 action，只用于查看 JSON 或给异常 run 写入
可审计原因。

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
export PYTHONPATH="$PWD/testbed${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 status

# 手柄按钮不可用时的等价单次 ARM；正常现场优先按一次手柄按钮 2
"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 arm-session

# 仅在没有 active run 时停止自动开始后续 run
"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 disarm-session

"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 intervention --reason "operator_intervention"

"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 abort --reason "operator_cancelled_run"

"${PYTHON}" -m testbed.cli.real_transition_control \
  --host 192.168.100.1 safety-stop --reason "hardware_safety_stop"
```

CLI 仍保留 `start-run/initial-ready/goal/dump-end/target-ready` 作为旧数据和协议诊断兼容
入口；正常录制只执行一次 `arm-session`，其余边界由数据自动生成。不要人工重复提交
goal/dump/target。

### 9.4 创建新 session

只在开始新的独立采集 context 时执行；同名目录不会被覆盖：

```bash
ssh slave-jetson
cd /media/mundane/D/Excavator_real_stack
export PYTHON="$PWD/.venv/bin/python"
export SESSION_ROOT="/media/mundane/EXTERNAL_USB/real_transition_raw_v2"

"${PYTHON}" -m testbed.cli.real_transition prepare-session \
  --output-root "${SESSION_ROOT}" \
  --session-id "rt_20260822_ctx04" \
  --seed 20260822
```

查看工具：

```bash
"${PYTHON}" -m testbed.cli.real_transition --help
"${PYTHON}" -m testbed.cli.real_transition prepare-session --help
"${PYTHON}" -m testbed.cli.real_transition verify-run --help
"${PYTHON}" -m testbed.cli.real_transition materialize-run --help
"${PYTHON}" -m testbed.cli.real_transition materialize-session --help
```

`init-home-calibration`、`capture-home-window`、`build-home-contract` 是 legacy/特殊回退
工具；当前规则型 ready 合同不需要 home/A/B 各 10 次采样。

### 9.5 CAN、频率、端口和日志

```bash
ip -details -statistics link show can2
ip -details -statistics link show can5
candump can2,18F021F6:1FFFFFFF

sudo nvpmodel -q
sudo jetson_clocks --show
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

ss -ltnp | grep -E ':(8765|8766|8770|8771)\b' || true
ps -ef | grep -E 'record_real|teleop_remote|policy_remote|bridge_gateway' \
  | grep -v grep

find /media/mundane/EXTERNAL_USB/real_transition_raw_v2 \
  -name run_manifest.json -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
  | sort
```

完整 v2 stack 中 `8765/8766/8770/8771` 都应监听；停止后都应释放。`8781` 是主端 UDP
状态镜像，不会出现在 Jetson TCP 监听列表。
