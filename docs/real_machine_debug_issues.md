# 真机调试问题记录

本文用于沉淀真机联调过程中遇到的问题、现场表现、根因判断、临时处理和长期修复方案。

记录原则：

- 不记录 sudo 密码、token、SSH 私钥等敏感信息。
- 每条问题必须包含：现象、影响范围、出现原因、触发场景、诊断命令、短期解决、长期解决、验证方式。
- 现场修复如果修改了系统配置，例如 `/etc/fstab`、udev、systemd service，需要记录修改内容和备份位置。

## 记录模板

```markdown
## YYYY-MM-DD 问题标题

### 现象

### 影响

### 出现原因

### 触发场景

### 诊断命令

### 短期解决

### 长期解决

### 验证方式

### 关联文件/配置
```

## 2026-07-02 GMSL 四路录制链路与 IMU/CAN 掉线记录

### 现象

切到 GMSL 四路相机后，从端 receiver/policy_remote 能启动并持续读到四路图像，
但真机链路反复进入 `fault`。日志中主要表现为：

- receiver health 报 `imu_missing:*`，常见组合包括 `imu_missing:0`、
  `imu_missing:0,1`、`imu_missing:0,3` 和短时 `imu_missing:0,1,2,3`。
- `can2` 曾出现 BUS-OFF；停止运行后再看实时 `candump can2` 可能为空，不能代表
  运行时没有 CAN 数据。
- `can3` 能持续看到 IMU 原始帧，且当前 latest raw capture 里 `can2/can3` 都有数据。
- 四路 GMSL 图像链路本身稳定，未看到图像预处理成为当前瓶颈。

### 影响

- IMU health 不稳定会阻止 receiver 进入可录制/可控制的健康状态；即使图像链路可用，
  HDF5 录制也不应在 IMU fault 状态下作为有效训练数据。
- 当前画面处理速度不是主要风险。更大的风险是 IMU/CAN 掉线导致 observation 中 qpos/qvel
  不可信。
- HDF5 保存是 episode 结束时同步写入，会有数秒 `saving` 间隔；但在线录制主循环不会因为
  每步 HDF5 写盘而被堵住。

### 出现原因

截至本记录，IMU 问题还不能断言为单一根因。已有证据支持：

- `can2` 内核日志在现场时间窗内有 BUS-OFF：
  `2026-07-02 17:23:47` 和 `2026-07-02 17:28:56`。
- receiver/policy steps 记录到 IMU online bit 反复变化，不是单纯因为当前进程已停止而看不到。
- `imu_qvel` 长日志显示四路 IMU 并非全程 online，其中 IMU 0、IMU 1 掉线占比高。
- 最新一轮 bridge 启动后，`candump -ta can2 can3` 能抓到两路原始 CAN 帧：
  60 秒内 `can2=16264` 帧，`can3=36240` 帧。

因此当前判断是：IMU/CAN 侧存在间歇性链路或设备健康问题，需要继续用原始 CAN、kernel journal、
receiver health 和 IMU/qvel 日志做对应分析。不要把问题归因到 GMSL 图像处理。

### 触发场景

- 从端执行 `scripts/slave_real_stack.sh run --force --policy-remote`，bridge 使用
  `--can-if can2 --imu-if can3`。
- IMU online/attitude health 丢失时，receiver 进入 `fault`。
- 如果 `can2` 进入 BUS-OFF，后续直接 `candump can2` 可能为空；需要先确认 link state，
  必要时重新 bring up 或重启现场 stack。

### 诊断命令

检查最新从端启动目录：

```bash
ssh slave-jetson '
cd /media/mundane/D/Excavator_real_stack
find artifacts/slave_stack -maxdepth 1 -mindepth 1 -type d \
  -printf "%T@ %TY-%Tm-%Td %TH:%TM:%TS %p\n" | sort -nr | head
'
```

检查 CAN 状态：

```bash
ssh slave-jetson 'ip -details -statistics link show can2; ip -details -statistics link show can3'
```

抓原始 CAN 数据，推荐输出为 CSV：

```bash
ssh slave-jetson '
cd /media/mundane/D/Excavator_real_stack
mkdir -p artifacts/imu_can_raw_csv_$(date +%Y%m%d_%H%M%S)
timeout 60s candump -ta can2 can3 > artifacts/imu_can_raw_csv_YYYYMMDD_HHMMSS/candump_can2_can3_ta_60s.log
'
```

过滤 kernel 里的 CAN 事件：

```bash
ssh slave-jetson '
journalctl --since "30 min ago" --no-pager |
grep -Ei "can|imu|bus|mttcan|bridge|receiver|fault|error" || true
'
```

### 短期解决

- IMU 未修好前，不要把 fault 状态下的 HDF5 当作有效训练数据。
- 若需要给硬件/固件侧分析，优先提供原始 CAN CSV、kernel journal、receiver log、
  `imu_qvel` JSONL/summary，而不是只提供当前空的 `candump`。
- 相机链路保持当前 GMSL grouped/JPEG 路径，不要回退到 full-resolution raw 或 RGBA 中间帧。
- `video4/video5` 的 `V4L2_BUF_FLAG_ERROR` 仍要统计，但已有动态画面和录制 benchmark 显示它没有
  直接阻断图像内容、CUDA 预处理或 SHM 发布。

### 长期解决

- 明确 IMU index、CAN raw address、物理安装位置和 CAN 接口的映射，避免把 `can2` 控制总线和
  `can3` IMU 诊断总线混用。
- 对 `can2` BUS-OFF 增加启动后状态检查和日志采集，必要时自动重启接口或在启动脚本中明确失败。
- 如果 IMU 掉线仍复现，做分层验证：
  1. 只跑 bridge，不跑相机，抓 `can2/can3` 原始 CSV。
  2. 跑 bridge + GMSL，不录 HDF5，检查 IMU health 是否变化。
  3. 跑完整 receiver/test-log，不开始有效 HDF5，检查 health 与 action/observation 时间戳。
  4. IMU 稳定后再做 10 秒真实 HDF5 保存验证。
- 后续如果要减少 episode 结束保存停顿，再把 HDF5 writer 改成异步 recorder；这不是当前阻塞主因。

### 验证方式

IMU/CAN 修复后的最低验收：

- `ip -details -statistics link show can2 can3` 均保持 `ERROR-ACTIVE`，没有新增 BUS-OFF。
- 60 秒 `candump -ta can2 can3` 转 CSV 后，接口和 CAN ID 计数稳定。
- receiver/test-log 中 `receiver_health_ok=1` 占比接近 100%，不再持续出现 `imu_missing:*`。
- `imu_qvel` 中 `imu_health.online` 四路长期为 `[1,1,1,1]`，host rx age p95 不异常。
- 四路 GMSL 仍满足 `drops=0`、`missing=0`、`cuda_fail=0`。
- HDF5 10 秒真实录制能完成保存，diagnostics 中四路 `image_timestamp_ns_*` 和 IMU/qpos/qvel 字段可追溯。

GMSL 图像链路已有性能证据：

- 最新 GMSL preprocess：四路 `drops=0`、`missing=0`、`cuda_fail=0`。
- CUDA preprocess kernel p95 约 `0.9 ms`。
- 最新 receiver/test-log：四路 image skew p95 `0.264 ms`，p99 `0.440 ms`。
- 3000 步 GMSL grouped benchmark：HDF5 `435 MB`，主循环 p95 `35.7 ms`，HDF5 save `3.7 s`。
- 外置 USB 当前 direct write 约 `104 MB/s`。

### 关联文件/配置

- 当前启动手册：`docs/host_slave_start_commands.md`
- 当前 GMSL policy/record config：`testbed/testbed/configs/policy_real_gmsl_four_camera_v1.yaml`
- GMSL preprocess manifest：`configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json`
- 旧日志分析包：`artifacts/imu_existing_logs_20260702_175348`
- 最新原始 CAN CSV：`artifacts/imu_can_raw_csv_20260702_180030/candump_can2_can3_ta_60s.csv`
- 最新原始 CAN summary：`artifacts/imu_can_raw_csv_20260702_180030/candump_can2_can3_ta_60s_summary.json`
- GMSL HDF5 benchmark：`artifacts/gmsl_recording_benchmark_grouped_20260701_153431/recording_benchmark_summary.json`

## 2026-05-28 摇杆映射和 go-home 方向排查记录

### 现象

真机启动手册曾混入 pygame 轴测试、手动响应记录和 go-home 调参过程。启动流程已精简，
这里只保留后续排查仍有价值的结论。

### 影响

如果引用了置换左右手柄前、左手柄 X/Y 旋转前的记录，可能误判 action 轴名、qpos 维度或
go-home 控制方向。

### 出现原因

2026-05-28 现场先确认 pygame 设备编号和轴方向，再把摇杆习惯调整为左手柄 `swing/stick`、
右手柄 `boom/bucket`，随后用手动动作和失败 HDF5 片段排查 go-home。

### 触发场景

- 修改 `axis_map/invert` 后，动作方向和实际挖机动作不一致。
- go-home 某个轴越控越远、卡在阈值附近，或看起来有液压死区/过冲。
- 需要解释某些旧参数为什么存在。

### 诊断命令

```bash
cd ~/Excavator_real_stack
rg -n "joystick_ids|axis_map|invert|control_signs|min_action|max_action" \
  testbed/testbed/configs/teleop_real_v1.yaml
```

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
./scripts/analyze_go_home_direction.py
```

### 短期解决

当前配置以 `testbed/testbed/configs/teleop_real_v1.yaml` 为准；下面只是历史排查线索：

- pygame 只读测试确认：左侧物理摇杆是 `joystick 0`，右侧物理摇杆是 `joystick 1`；
  `axis 0` 左推为负、右推为正，`axis 1` 前推为负、后拉为正。
- 当前最终摇杆配置来自现场修正：`axis_map: [1, 1, 0, 0]`、
  `invert: [true, false, true, false]`。
- `artifacts/manual_response/20260528_145235.jsonl` 是置换左右手柄后较有参考价值的一次记录：
  swing/boom/stick/bucket 分别和 qvel `[0,1,2,3]` 同名相关，bucket 只确认了正负同名关系，
  收斗/开斗语义仍应现场确认。
- `artifacts/manual_response/20260528_150222.jsonl` 用于估计液压死区：当时大致认为 swing 需要
  `0.5+` 才明显响应，boom 约 `0.35-0.45`，stick 约 `0.42-0.55`，bucket 约 `0.50-0.60`。
- `recommendation=flip` 表示该轴在 go-home 段主动下发时 `|error|` 平均增大，需要检查或翻转
  `control_signs`。
- 曾经为 go-home 功能测试把 `near_tolerance_rad` 临时设为 `[999.0, 999.0, 999.0, 999.0]`；
  这个只代表关闭 near-home 启动门禁做测试，不代表允许任意位置自主回 home。

### 长期解决

- 启动手册只保留当前可执行流程和当前映射。
- 每次现场改 `axis_map/invert/control_signs/min_action/max_action` 后，重新做短的单轴验证。
- 最终参数只以当前配置文件为准，本文档只解释历史来源。

### 验证方式

- `docs/host_slave_start_commands.md` 不应再出现长段历史调参记录。
- 当前映射表应明确物理方向、pygame 原始正负、`action[i]` 正负和实际动作。
- 真机变更后，确认 qpos/qvel 与预期一致再更新文档。

### 关联文件/配置

- 当前现场流程：`docs/host_slave_start_commands.md`
- 当前真机控制配置：`testbed/testbed/configs/teleop_real_v1.yaml`
- 历史手动响应：`artifacts/manual_response/20260528_145235.jsonl`
- 历史手动响应：`artifacts/manual_response/20260528_150222.jsonl`

## 2026-05-26 Jetson 1TB NVMe 存在但 `/media/mundane/D` 为空

### 现象

在主端 VS Code Remote-SSH 连接从端 Jetson 后，无法在原来的 `/media` 路径下找到仓库。
`/media/mundane/D` 目录存在，但目录内容为空，看起来像 1TB 硬盘丢失。

本次现场确认到的正确仓库路径是：

```text
/media/mundane/D/Excavator_real_stack
```

### 影响

- VS Code Remote-SSH 无法打开原本位于 1TB 盘上的仓库。
- 真机启动手册中的从端命令无法执行，因为命令默认 `cd /media/mundane/D/Excavator_real_stack`。
- 如果误以为仓库丢失，可能会在 `/home` 重新 clone 或切换到错误代码版本，导致主从端代码不一致。

### 出现原因

1TB NVMe 盘本身没有丢失，内核已经识别到设备：

```text
/dev/nvme0n1
```

该设备是 ext4 文件系统，label 为 `D`，UUID 为：

```text
8f5abb03-6257-4003-b99f-731521214775
```

问题的直接原因是：设备存在，但没有被挂载到 `/media/mundane/D`。

更具体地说：

- `/media/mundane/D` 只是一个挂载点目录。磁盘没有挂载时，这个目录本身会显示为空。
- Jetson 作为从端常通过 SSH/VS Code Remote-SSH 使用，远程非图形会话不会可靠触发桌面环境的自动挂载。
- `udisksctl mount` 在非交互 SSH 会话中可能需要 polkit 授权；没有可用的图形/TTY 认证代理时会失败。
- 原先 `/etc/fstab` 没有为该 1TB NVMe 配置开机自动挂载项，所以重启或异常掉电后不会自动回到 `/media/mundane/D`。

### 触发场景

常见触发条件：

- Jetson 重启、断电恢复或桌面用户未登录。
- 只通过 VS Code Remote-SSH 登录，没有在 Jetson 本机桌面文件管理器中点开磁盘。
- `/etc/fstab` 中没有该盘的 UUID 挂载配置。
- `/media/mundane/D` 目录存在，导致误判为“盘已挂载但数据没了”；实际只是空挂载点。

### 诊断命令

在主端执行，检查 Jetson 是否识别到磁盘：

```bash
ssh slave-jetson \
  'lsblk -e7 -o NAME,PATH,SIZE,TYPE,FSTYPE,LABEL,UUID,MOUNTPOINTS,MODEL'
```

重点看：

```text
nvme0n1 /dev/nvme0n1 953.9G disk ext4 D 8f5abb03-6257-4003-b99f-731521214775
```

检查挂载状态：

```bash
ssh slave-jetson 'findmnt /media/mundane/D || true'
```

检查挂载点是否只是空目录：

```bash
ssh slave-jetson 'ls -la /media/mundane/D'
```

检查仓库是否可见：

```bash
ssh slave-jetson \
  'find /media/mundane/D -maxdepth 5 -type d -name Excavator_real_stack 2>/dev/null'
```

### 短期解决

手动挂载 1TB NVMe：

```bash
ssh slave-jetson
sudo mkdir -p /media/mundane/D
sudo mount /dev/disk/by-label/D /media/mundane/D
```

挂载后确认仓库：

```bash
findmnt /media/mundane/D
ls -la /media/mundane/D
cd /media/mundane/D/Excavator_real_stack
git status --short --branch
```

如果 label 改过或不确定，优先用 UUID：

```bash
sudo mount /dev/disk/by-uuid/8f5abb03-6257-4003-b99f-731521214775 /media/mundane/D
```

### 长期解决

把 1TB NVMe 写入 Jetson 的 `/etc/fstab`，让系统启动时自动挂载：

```text
UUID=8f5abb03-6257-4003-b99f-731521214775 /media/mundane/D ext4 defaults,nofail,x-systemd.device-timeout=10 0 2
```

本次现场已经完成该配置，并在修改前备份：

```text
/etc/fstab.codex-backup-20260526-105515
```

配置说明：

- `UUID=...`：用文件系统 UUID 定位磁盘，避免 `/dev/nvme0n1` 名称变化带来的风险。
- `/media/mundane/D`：保持原现场路径不变，避免启动脚本和文档大面积调整。
- `ext4`：该 1TB NVMe 的文件系统类型。
- `nofail`：即使磁盘临时不可用，也不阻塞系统启动。
- `x-systemd.device-timeout=10`：设备等待超时 10 秒，避免开机长时间卡住。

### 验证方式

立即验证：

```bash
findmnt /media/mundane/D -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -e7 -o NAME,PATH,SIZE,TYPE,FSTYPE,LABEL,UUID,MOUNTPOINTS
git -C /media/mundane/D/Excavator_real_stack rev-parse --show-toplevel
git -C /media/mundane/D/Excavator_real_stack status --short --branch
```

预期结果：

```text
/media/mundane/D mounted from /dev/nvme0n1
/media/mundane/D/Excavator_real_stack
```

检查 `/etc/fstab` 语法：

```bash
findmnt --verify --verbose
```

本次验证结果：

```text
0 parse errors, 0 errors
```

重启后验证：

```bash
sudo reboot
```

重新 SSH 登录后执行：

```bash
findmnt /media/mundane/D
ls /media/mundane/D/Excavator_real_stack
```

### 关联文件/配置

- Jetson 从端：SSH `slave-jetson`，控制链路 `192.168.100.1`
- 1TB NVMe：`/dev/nvme0n1`
- 挂载点：`/media/mundane/D`
- 仓库路径：`/media/mundane/D/Excavator_real_stack`
- 系统配置：`/etc/fstab`
- fstab 备份：`/etc/fstab.codex-backup-20260526-105515`
- 相关现场命令：`docs/host_slave_start_commands.md`
