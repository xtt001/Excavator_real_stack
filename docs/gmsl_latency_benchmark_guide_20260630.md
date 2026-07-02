# GMSL 四路相机时延测试操作指南

日期：2026-06-30

本文用于现场侧准备 GMSL 相机处理时延测试。目标是先得到可复查的分段数据，再决定是否继续投入
GStreamer/NVMM/DMABUF 或自定义 CUDA kernel。

## 1. 测试口径

本轮 benchmark 的处理时延口径是：

```text
相机 buffer dequeue / cap.read 返回 -> 图像预处理输出 ready
```

不包含：

- 相机曝光等待和帧周期。
- `imshow` 显示。
- 视频写盘。
- ACT policy forward。
- 控制下发。

因此不要把 30 fps 相机天然的 33.3 ms 帧周期混进这轮 4 ms 处理预算。这里先测每帧进入进程后的处理成本。

## 2. 拉取代码并确认设备列表

现场侧进入仓库并拉取当前分支：

```bash
cd /media/mundane/D/Excavator_real_stack
git pull
```

先确认 bring-up 设备列表。当前默认仍是 `video6 video7`：

```bash
GMSL_PRINT_CONFIG_ONLY=1 GMSL_VIDEO_DEVICES="6 7" scripts/bring_up_gmsl_cameras.sh
```

四路测试时把 `GMSL_VIDEO_DEVICES` 改成现场真实编号。当前现场四路为 `4 5 6 7`：

```bash
GMSL_PRINT_CONFIG_ONLY=1 GMSL_VIDEO_DEVICES="4 5 6 7" scripts/bring_up_gmsl_cameras.sh
```

确认后再执行真实 bring-up：

```bash
GMSL_VIDEO_DEVICES="4 5 6 7" scripts/bring_up_gmsl_cameras.sh
```

## 3. 编译 benchmark

```bash
cmake -S tools/gmsl_latency_benchmark -B build/gmsl_latency_benchmark
cmake --build build/gmsl_latency_benchmark -j
```

如果 CMake 找不到 OpenCV C++ 开发包，先在 Jetson 环境中修复 OpenCV dev 安装或指向正确的
`OpenCV_DIR`。这个 benchmark 需要 OpenCV C++，不是 Python `cv2`。

## 4. capture-only 基线

先跑 capture-only，判断四路同时 `cap.read` 是否已经有明显阻塞或队列问题。

两路当前相机：

```bash
mkdir -p artifacts/gmsl_latency

build/gmsl_latency_benchmark/gmsl_latency_benchmark \
  --capture-only \
  --raw-camera video6=/dev/video6 \
  --raw-camera video7=/dev/video7 \
  --frames 300 \
  --warmup 30 \
  --output-json artifacts/gmsl_latency/capture_only_video6_video7.json
```

四路相机：

```bash
mkdir -p artifacts/gmsl_latency

build/gmsl_latency_benchmark/gmsl_latency_benchmark \
  --capture-only \
  --raw-camera video4=/dev/video4 \
  --raw-camera video5=/dev/video5 \
  --raw-camera video6=/dev/video6 \
  --raw-camera video7=/dev/video7 \
  --frames 300 \
  --warmup 30 \
  --output-json artifacts/gmsl_latency/capture_only_four_camera.json
```

## 5. 当前 384x216 预处理去畸变基线

当前四路 H190TA 内参都已导入。先按
`configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json`
跑完整连续采集和 384x216 virtual rectilinear remap 分段计时：

```bash
mkdir -p artifacts/gmsl_latency

build/gmsl_latency_benchmark/gmsl_latency_benchmark \
  --preprocess-manifest configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json \
  --camera video7=/dev/video7 \
  --camera video6=/dev/video6 \
  --camera video4=/dev/video4 \
  --camera video5=/dev/video5 \
  --frames 300 \
  --warmup 30 \
  --output-json artifacts/gmsl_latency/current_preprocess_four_camera.json
```

如果要强制 CPU 路径做对照：

```bash
build/gmsl_latency_benchmark/gmsl_latency_benchmark \
  --cpu \
  --preprocess-manifest configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json \
  --camera video7=/dev/video7 \
  --camera video6=/dev/video6 \
  --camera video4=/dev/video4 \
  --camera video5=/dev/video5 \
  --frames 300 \
  --warmup 30 \
  --output-json artifacts/gmsl_latency/current_preprocess_four_camera_cpu.json
```

## 6. 汇总结果

JSON 每路会有这些字段：

- `read_ms`：`cap.read` 耗时。
- `color_ms`：UYVY/gray 到 BGR 的 CPU 转换耗时。
- `upload_ms`：CPU 到 GPU 上传耗时，CPU 路径为 0。
- `remap_ms`：OpenCV fisheye remap 耗时。
- `download_ms`：GPU 到 CPU 下载耗时，CPU 路径为 0。
- `rotate_ms`：按 manifest 做 180 度旋转的耗时。
- `process_ms`：颜色转换到 remap/rotate 输出完成的总处理耗时。
- `frame_ms`：本 benchmark 对这一帧的总循环耗时。

重点看每路 `p50/p95/p99/max`，不要只看平均值。4 ms 目标建议先以 `process_ms p95`
为硬指标候选，同时记录 `frame_ms p95` 判断 capture 是否拖慢了整体循环。

跑完一组或多组 benchmark 后，用汇总脚本生成 Markdown 和 CSV：

```bash
python3 tools/gmsl_latency_benchmark/summarize_latency_json.py \
  artifacts/gmsl_latency/capture_only_four_camera.json \
  artifacts/gmsl_latency/current_preprocess_four_camera.json \
  --output-markdown artifacts/gmsl_latency/summary.md \
  --output-csv artifacts/gmsl_latency/summary.csv \
  --process-p95-budget-ms 4.0
```

如果 capture-only 也要设置硬预算，可以额外传：

```bash
--frame-p95-budget-ms 4.0
```

默认情况下，非 capture-only 结果按 `process_ms p95 <= 4.0 ms` 标记 `PASS` / `FAIL`；
capture-only 结果在未设置 `--frame-p95-budget-ms` 时标记为 `INFO`，用于判断输入路径是否值得继续优化。

## 7. 初步判断规则

- 如果 capture-only 的 `frame_ms p95` 已明显高于 4 ms，先排查 V4L2/GStreamer 输入路径、
  buffer 数、同步触发、分辨率和驱动队列。
- 如果 capture-only 很低，但 OpenCV remap 的 `upload_ms + remap_ms + download_ms`
  明显高于 4 ms，说明当前 OpenCV CPU/GPU 往返路径不适合生产。
- 如果 `download_ms` 占比大，生产路径应避免下载回 CPU，直接输出 policy tensor。
- 如果 `read_ms max` 偶发很高，保留原始 JSON，结合 `tegrastats` 判断是否有温度、频率或队列抖动。

生产化优化路线见：

```text
docs/gmsl_realtime_preprocessing_optimization_plan_20260630.md
```

## 8. 同步记录系统状态

跑 benchmark 时另开终端记录 Jetson 状态：

```bash
tegrastats --interval 1000 | tee artifacts/gmsl_latency/tegrastats.txt
```

同时保存：

```bash
v4l2-ctl --all -d /dev/video6 > artifacts/gmsl_latency/video6_v4l2_all.txt
v4l2-ctl --all -d /dev/video7 > artifacts/gmsl_latency/video7_v4l2_all.txt
```

四路时对每一路都保存一份 `v4l2-ctl --all`。这些文件和 JSON 一起用于后续判断是否需要
GStreamer/NVMM、VPI/NPP 或 fused CUDA kernel。

## 9. 当前时间戳来源

需要记录相机帧时间戳时，优先使用直接 V4L2 `ioctl` 路径，不要依赖 OpenCV
`VideoCapture` 或 libv4l 包装层。当前仓库提供 probe：

```bash
build/gmsl_latency_benchmark/v4l2_timestamp_probe \
  --device /dev/video4 \
  --frames 120 \
  --warmup 8 \
  --buffers 4 \
  --output-json artifacts/gmsl_latency/video4_timestamp.json
```

2026-07-01 在当前 SG8A + SG3S-ISX031 四路现场配置上，直接 `VIDIOC_DQBUF`
看到的结果是：

- `/dev/video4`、`/dev/video5`、`/dev/video6`、`/dev/video7` 的
  `timestamp_clock = monotonic`。
- `timestamp_source = eof`，即当前用户态拿到的是 V4L2/VI buffer 的 EOF
  时间戳，不是已经证实的相机曝光开始时间戳。
- 四路 timestamp interval p50/p95 都为 `33.334 ms`，与 30 fps 触发周期一致。
- `video4`、`video5` 当前 buffer flags 带 `V4L2_BUF_FLAG_ERROR`，但 sequence
  和 timestamp 连续；这需要作为 QC 字段记录，不能直接忽略。
- SG8A 驱动包设备树里当前各 sensor mode 的 `embedded_metadata_height = "0"`，
  所以现有图像流没有额外 embedded metadata 行可解析相机侧时间戳。

因此，短期 HDF5/在线队列应记录：

```text
camera_key, serial, device, v4l2_sequence, v4l2_timestamp_ns,
v4l2_timestamp_clock, v4l2_timestamp_source, v4l2_flags,
host_arrival_mono_ns
```

如果后续需要更精确的曝光开始时间戳，优化方向是：

- 向森云确认 SG3S-ISX031 / M3A 对应型号是否支持 embedded metadata 或寄存器形式的
  frame counter / exposure timestamp。
- 若支持，修改设备树和驱动，使 `embedded_metadata_height > 0` 并在用户态保存/解析。
- 若只能依赖 Jetson VI timestamp，继续查 Tegra VI 驱动 timestamp source，确认是否能切到
  SOF 或记录 `descr->status.sof_timestamp`。
