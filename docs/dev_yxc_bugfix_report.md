# dev_yxc Bugfix 记录

**日期**：2026-05-18
**分支**：`dev_yxc`

本记录说明本轮只针对 `dev_yxc` 分支做的小范围 bug 修复；未触碰真机硬件、ROS 节点或 CAN 通信。

## 修复 1：`--data-side` 默认值覆盖配置

### 问题

`tb-record-real` 的 `--data-side` 参数在 argparse 中默认写死为 `slave`。这样即使用户在 `teleop_real_v1.yaml` 中设置 `real.data_side: host`，或通过环境变量 `EXCAVATOR_DATA_SIDE=host` 指定主端落盘，也会被命令行默认值覆盖。

### 影响

- 主端/从端录制部署切换不符合文档预期。
- `EXCAVATOR_DATA_SIDE` 实际无法作为默认部署开关使用。
- 同事在主端录制、从端 gateway 的模式下容易误写到从端默认路径。

### 修改

- 将 `testbed/testbed/cli/record_real.py` 中 `--data-side` 的 argparse 默认值从 `slave` 改为 `None`。
- 保留 `testbed.cli.data_side.resolve_data_side()` 的真实默认链路：
  1. 命令行 `--data-side`
  2. YAML `real.data_side`
  3. 环境变量 `EXCAVATOR_DATA_SIDE`
  4. 默认 `slave`
- 新增单测覆盖 YAML、环境变量、CLI 覆盖优先级。

## 修复 2：`dev_yxc_changelog` 与当前代码状态不一致

### 问题

`docs/dev_yxc_changelog.md` 仍保留推送前描述，例如：

- 多处写着“未提交”。
- 说明 C++ bridge 未实现 `send_status`。
- 写法 `send_status(status11)` 与当前协议实际 payload `toggle_mask` 不一致。

### 影响

会误导后续联调：当前 C++ bridge 已有 `send_status.request(toggle_mask)`，gateway/testbed 也是通过 `toggle_mask` 发送 status 上升沿。

### 修改

- 将“未提交”类描述调整为“已合入”。
- 将 `send_status(status11)` 改为 `send_status(toggle_mask)`。
- 将后续待办从“C++ bridge 增加 send_status”改为“真机确认 status 位到液压控制/安全逻辑的映射语义”。

## 修复 3：文档行尾空格

### 问题

`git diff --check origin/main..HEAD` 报告文档中存在 trailing whitespace。

### 修改

- 清理 `docs/dev_yxc_changelog.md` 与 `docs/realworld_host_slave_runbook.md` 中的行尾空格。

## 修复 4：gateway 因 `read_state` 业务失败反复断开 C++ bridge

### 问题

实际真机测试中，C++ bridge 日志出现 `client connected/disconnected` 反复刷屏，并伴随 `watchdog forced zero command after ... ms without valid action`。

代码检查发现：Python gateway 转发请求时使用 `JsonTcpBridgeClient._request()`。该方法会在上游返回 `ok=false` 时抛出 `BridgeProtocolError`。gateway 把这个异常当成连接/协议错误处理，随后主动 `force_close()` upstream。于是只要 C++ bridge 的 `read_state` 因状态暂时超时返回一次 `ok=false`，gateway 就会断开到 C++ bridge 的长连接，下一次请求再重连。

### 影响

- C++ bridge 端看到频繁 `client connected/disconnected`。
- 上游连接重建增加控制周期抖动。
- 如果 `send_action` 间隔超过 bridge watchdog，例如 200ms，底层会强制零命令。

### 修改

- 在 `JsonTcpBridgeClient` 中新增 `_request_response()`，返回完整 response，包括 `ok/error/payload`，但不因 `ok=false` 断开连接。
- gateway 改用 `_request_response()` 转发上游响应：
  - socket 断开、协议错误仍会丢弃 upstream 并重连。
  - C++ bridge 返回的业务失败会原样透传给 testbed，不再主动断开 upstream。
- 为 TCP client 增加请求锁，避免后续控制保活线程与主循环共用 client 时发生读写交错。
- 新增单测验证：一次 `read_state.response(ok=false)` 后，同一 TCP 连接仍可继续发送 `send_action`。

### 仍需注意

这次修的是“业务失败导致 gateway 断连”的 bug。控制 heartbeat 与录制/观测链路耦合的问题见下方“修复 5”。

## 修复 5：控制 heartbeat 与录制/观测链路耦合

### 问题

真机底层是速度控制，低层 watchdog 需要持续收到速度命令。旧录制循环把控制、状态读取和数据缓存串在同一线程里：

```text
joystick -> send_action -> read_state -> recorder.record -> sleep
```

因此即使摇杆保持不动，只要 `read_state`、图像处理、Python 数组拷贝或后续 HDF5 缓存变慢，下一次 `send_action` 就会晚到，最终触发 bridge watchdog。

### 修改

- 新增 `testbed.backends.real.action_pump.RealActionPump`：
  - 后台线程按固定频率重复发送“最后一次安全 action”。
  - `update_action()` 更新最新速度命令，默认立即发送一次以降低手柄响应延迟。
  - `stop()` 时默认发送零命令，避免退出后保持上一次速度。
- `tb-record-real` 在 `backend=bridge_tcp` 且 `real.control_pump.enabled=true` 时启用 action pump：
  - 控制链路按 `real.control_pump.hz` 发送。
  - 录制/观测循环按 `task.record_hz` 采样。
  - status toggle 也走 control pump 的控制连接。
- C++ `excavator_real_bridge` 新增后台状态缓存线程：
  - `ExcavatorReceive::get()` 在后台持续取状态。
  - `read_state.request` 只读取缓存，不再阻塞主 TCP 请求循环。
  - 返回 payload 增加 `snapshot_age_ms` 和 `state_loop_tick`，便于后续判断状态是否滞后。

### 效果

- 摇杆不动时，最后一次速度命令仍会按 50Hz 持续发送。
- 录制/图像链路慢时，不会直接拖住控制 heartbeat。
- `read_state` 暂时慢或无新状态时，不会长时间占住 C++ bridge 的请求处理线程。

### 限制

- action pump 当前推荐通过 Python gateway `8765` 使用。不要让 testbed 的控制连接和状态连接同时直连 C++ bridge `8766`，因为 C++ bridge 当前仍是单 client 处理模型。
- 真机端仍需验证 `snapshot_age_ms` 的合理范围，并据此决定是否把 stale 状态写入 safety guard。

## 验证范围

按照本轮要求，未运行完整测试套件。建议只做轻量验证：

- 针对 `data_side` 相关单测。
- 针对 TCP client 上游业务失败保连接的单测。
- 针对 action pump 固定频率重复发送最新 action 的单测。
- `git diff --check`。
- Python 编译检查相关改动文件。
