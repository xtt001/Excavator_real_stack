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
ssh mundane@192.168.31.170 \
  'lsblk -e7 -o NAME,PATH,SIZE,TYPE,FSTYPE,LABEL,UUID,MOUNTPOINTS,MODEL'
```

重点看：

```text
nvme0n1 /dev/nvme0n1 953.9G disk ext4 D 8f5abb03-6257-4003-b99f-731521214775
```

检查挂载状态：

```bash
ssh mundane@192.168.31.170 'findmnt /media/mundane/D || true'
```

检查挂载点是否只是空目录：

```bash
ssh mundane@192.168.31.170 'ls -la /media/mundane/D'
```

检查仓库是否可见：

```bash
ssh mundane@192.168.31.170 \
  'find /media/mundane/D -maxdepth 5 -type d -name Excavator_real_stack 2>/dev/null'
```

### 短期解决

手动挂载 1TB NVMe：

```bash
ssh mundane@192.168.31.170
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

- Jetson 从端：`192.168.31.170`
- 1TB NVMe：`/dev/nvme0n1`
- 挂载点：`/media/mundane/D`
- 仓库路径：`/media/mundane/D/Excavator_real_stack`
- 系统配置：`/etc/fstab`
- fstab 备份：`/etc/fstab.codex-backup-20260526-105515`
- 相关现场命令：`docs/host_slave_start_commands.md`
