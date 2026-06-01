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
