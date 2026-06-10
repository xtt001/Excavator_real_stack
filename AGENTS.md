# AGENTS.md

本文件定义本仓库的全局代理工作规则。除非更深层目录中的
`AGENTS.md` 明确覆盖，否则这些规则适用于整个仓库。

## 全局规则

1. 用户可见回复必须使用中文，除非用户明确要求使用其他语言。

2. 如果任务中有任何不确定信息，必须先向用户提问确认。不要自行假设
   用户意图、技术细节、运行环境、数据含义或实现方向。

## 当前主从现场环境

1. 当前这台开发电脑是主端。主端主要负责代码编辑、主端 sender/手柄控制、
   rqt/FPV 查看、QC 分析和从端运维；不要在主端启动真机 C++ bridge，也不要把
   训练 HDF5 直接写到主端当作现场默认流程。

2. Jetson 是从端。SSH 入口使用 `slave-jetson`，当前配置为
   `mundane@192.168.100.1`；非 SSH 的 TCP 控制链路也使用 `192.168.100.1`。
   SSH 只作为登录、启动脚本、查日志、挂载磁盘、同步文件和诊断通道，不承担
   50Hz 实时控制。

3. 从端仓库路径是 `/media/mundane/D/Excavator_real_stack`。该路径位于 Jetson
   的 1TB NVMe 挂载点 `/media/mundane/D` 下；如果路径为空或仓库不可见，应先检查
   从端磁盘挂载状态，而不是在 `/home` 重新 clone 或切换到另一份代码。

4. 每次在主端完成代码更新，并且需要在真机、从端脚本、从端配置或 Jetson 运行链路
   上生效时，必须把相关仓库文件同步到从端
   `/media/mundane/D/Excavator_real_stack`。默认直接同步相关文件到从端 Jetson，
   不需要再次提示用户；不要自行使用带破坏性的同步语义覆盖或删除从端内容。

5. 从端负责 real CAN bridge、Orbbec 相机、FPV 到 SHM、gateway、
   `policy_remote`/receiver、本地 USB 写盘和 IMU/CAN 只读诊断。主端负责
   `teleop_remote` 手柄 sender、rqt 看图、在线 QC watcher 和离线分析。

6. 固定端口语义必须保持一致：从端 gateway 为 `127.0.0.1:8765`，从端 C++ bridge
   为 `127.0.0.1:8766`，主端 sender 连接从端 receiver 为
   `192.168.100.1:8770`。

7. 录制数据、IMU/qvel 日志和 policy control 测试日志默认都在从端外置 USB 或从端
   `/media/mundane` 相关路径下：
   `/media/mundane/EXTERNAL_USB/real_teleop_v1`、
   `/media/mundane/EXTERNAL_USB/imu_qvel_tests`、
   `/media/mundane/EXTERNAL_USB/policy_control_tests`。
   分析这些文件时，应通过 SSH/rsync 拉取已完成文件到主端缓存，或在确认路径后直接
   从从端读取；不要误以为这些日志默认存在于主端仓库目录。

8. 当前现场启动、停止、主从职责和 QC 命令以
   `docs/host_slave_start_commands.md` 为准；问题复盘和架构说明参考
   `docs/real_stack_problem_solution_summary_20260608.md` 与
   `docs/real_machine_debug_issues.md`。

## 默认排查和修改原则

1. 默认采用最小修改：只改当前问题直接相关的文件，不做无关重构、风格调整、
   配置清理或文档改写。

2. 处理问题前先找根因，不先猜测式修补。排查顺序为：先确认现象；再查
   `docs/`、历史复盘、脚本和配置，判断过去是否出现过；如果过去出现过，优先
   按已有结论处理并验证；如果过去没有出现过，再分析现在出现的原因。

3. 新问题优先检查最近变化，包括当前 `git status --short`、相关 `git diff`、
   最近配置变更、主从端代码是否同步、运行环境变化、磁盘挂载、端口占用和
   数据路径变化。涉及从端时，先确认需要生效的主端改动是否已经同步到
   `/media/mundane/D/Excavator_real_stack`。

4. 同步主端改动到从端时，默认直接同步相关文件到 Jetson，不需要再次提示用户。
   整仓库同步、带 `--delete` 的同步、覆盖从端未提交修改或删除从端文件前，必须
   先得到用户明确确认。同步后应校验文件哈希、`git status` 或其他可验证结果。

5. 只有用户明确表达需要“沉淀”、更新文档、写复盘或整理经验时，才把问题处理过程
   写入文档。需要沉淀时，按“现象 -> 影响 -> 根因 -> 修改 -> 验证 -> 复用经验”
   组织内容；现场启动和操作变化优先更新 `docs/host_slave_start_commands.md`。
