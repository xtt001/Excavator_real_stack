# GMSL 四路实时预处理优化计划

日期：2026-06-30

目标：先用现有 OpenCV/V4L2 benchmark 得到可复查的四路分段时延，再决定是否进入
GStreamer/NVMM/DMABUF、VPI/NPP 或 fused CUDA kernel 实现。目标预算是四路相机图像进入进程后，
预处理输出 ready 的 `process_ms p95 <= 4 ms`。

## 1. 不先写生产 CUDA 的原因

当前还缺少四路现场实测：

- 四路同时 capture-only 的 `read_ms` / `frame_ms`。
- 两路或四路 OpenCV remap 的 `upload_ms` / `remap_ms` / `download_ms`。
- Jetson 现场 OpenCV 是否实际启用 CUDA `cudawarping`。
- `tegrastats` 下 GPU/CPU 频率、温度和偶发抖动。

没有这些数据时直接写 fused CUDA kernel，容易把时间花在错误瓶颈上。如果瓶颈主要在
V4L2 dequeue、驱动队列、格式转换或 CPU/GPU 往返，单独优化 remap kernel 不一定能把总时延压到
4 ms 内。

## 2. 现场先跑的三组数据

1. capture-only 四路：
   判断输入路径、驱动队列和多路并发 `cap.read` 是否稳定。
2. OpenCV remap 两路：
   用当前已有内参的 `video6` / `video7` 形成去畸变 baseline。
3. OpenCV remap 四路：
   等 `eye_left` / `eye_right` 内参导入后再跑。

每次都保存 benchmark JSON、`tegrastats` 和每路 `v4l2-ctl --all`。

## 3. 汇总和决策口径

用汇总脚本把多个 JSON 合成一页 Markdown 和一份 CSV：

```bash
python3 tools/gmsl_latency_benchmark/summarize_latency_json.py \
  artifacts/gmsl_latency/capture_only_four_camera.json \
  artifacts/gmsl_latency/opencv_remap_video6_video7.json \
  --output-markdown artifacts/gmsl_latency/summary.md \
  --output-csv artifacts/gmsl_latency/summary.csv \
  --process-p95-budget-ms 4.0
```

重点看：

- `read_ms p95` / `frame_ms p95`：输入路径是否已经超过预算。
- `upload_ms p95 + download_ms p95`：CPU/GPU 往返是否主导。
- `remap_ms p95`：纯 remap 是否值得单独优化。
- `process_ms p95`：当前预处理链路是否满足 4 ms 目标。

## 4. 进入下一阶段的门槛

如果 capture-only 的 `frame_ms p95` 已经高于 4 ms：

- 优先检查 GStreamer/NVMM 输入、buffer 数、同步触发、分辨率、像素格式和驱动队列。
- 这时不要先写 remap CUDA kernel，因为输入端已经超预算。

如果 capture-only 稳定，但 OpenCV GPU 的 `upload_ms + download_ms` 占比高：

- 优先做 GStreamer/NVMM/DMABUF 零拷贝输入。
- 目标是让图像留在 GPU 或可被 CUDA 直接访问的 buffer 中。

如果 `remap_ms p95` 是主要瓶颈：

- 再考虑 fused CUDA kernel，把 fisheye remap、resize、rotate、RGB/normalize 合并。
- 输出不应下载回 CPU，而应直接写 policy tensor 或训练/推理约定的 GPU buffer。

如果 OpenCV remap 四路 `process_ms p95 <= 4 ms` 且 `frame_ms p95` 稳定：

- 可以先保留 OpenCV 路径作为短期工程 baseline。
- 仍需保留 JSON 和系统状态作为回归基线，避免后续相机数、分辨率或模型输入变化后失控。

## 5. 预计算 remap map

无论短期继续用 OpenCV，还是后续切到 fused CUDA，都应把 per-camera remap map 预计算列为正式待办。
运行时不应每帧从内参重新解算 fisheye 几何，而应在初始化阶段读取或生成一次 map，实时循环只做查表采样。

建议产物：

```text
configs/camera_remap/gmsl_h190ta_four_camera/
  remap_manifest.json
  stick_top_map_float32.bin
  stick_bottom_map_float32.bin
  eye_left_map_float32.bin
  eye_right_map_float32.bin
```

`remap_manifest.json` 至少记录：

- `mount_position`, `camera_key`, `serial`, `intrinsics_file`
- `output_width`, `output_height`, `hfov_deg`, `pitch_down_deg`, `orientation`
- `map_format`, `map_file`, `map_sha256`
- `source_intrinsics_manifest_sha256`, `source_preprocess_manifest_sha256`

收益：

- 把 fisheye 反投影、虚拟视角、pitch/orientation 等几何计算移出逐帧路径。
- 训练、推理、benchmark 可以引用同一个 map hash，防止 transform 版本漂移。
- OpenCV baseline 可以读取同一份 map；后续 CUDA kernel 也可以复用同一份采样表。
- fused CUDA 可把 `remap + resize + rotate + normalize + layout transform` 合成一次 kernel，并直接输出 policy tensor。

失效规则：

- 任一路内参、输出尺寸、HFOV、pitch、yaw、roll、orientation 或相机顺序变化，必须重新生成 map。
- 训练数据 manifest、runtime config、latency summary 都应记录所用 `remap_manifest` hash。
- 每次 map 更新后，重新生成 contact sheet 和 latency summary。

## 6. fused CUDA 目标接口

生产实现建议只在数据证明需要后再写，目标接口保持窄：

```text
input:  four camera frames in fixed training order
maps:   per-camera fisheye sampling maps / virtual-view parameters
output: policy tensor in fixed camera order, fixed size, normalized layout
```

必须保持这些不变量：

- 训练和推理使用同一相机顺序：`stick_top`, `stick_bottom`, `eye_left`, `eye_right`。
- 每路 orientation、pitch、HFOV 和 resize 版本必须来自 manifest。
- `31460/stick_top` 保留 `rotate_180`，`31459/stick_bottom` 保留 `pitch_down_deg=20`，除非现场证据更新。
- 每次 kernel 或输入路径变更后，重新生成 contact sheet 和 latency summary。

## 7. 外参和几何融合

外参求解放在安装结构固定、棋盘格多姿态证据齐全之后。短期 ACT 输入不依赖高精度多相机几何融合，
但需要工程化记录每路安装位置、orientation、pitch、内参和证据目录。

当需要进一步做多相机几何融合、ROI 随外参定义、或安装漂移诊断时，再基于
`configs/camera_calibration/gmsl_h190ta_four_camera/extrinsics_template.json` 生成正式外参 manifest。
