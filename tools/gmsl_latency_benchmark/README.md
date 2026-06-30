# GMSL Latency Benchmark

Standalone GMSL camera latency benchmark for the H190TA fisheye cameras. It is
not part of the production runtime path. Use it to measure the current OpenCV
V4L2 preprocessing baseline before investing in GStreamer/NVMM or a fused CUDA
kernel.

The measured latency boundary is:

```text
cap.read returned buffer -> preprocessing output ready
```

Display, video recording, policy forward, and control send are intentionally
excluded.

## Build

Build on the Jetson or another machine with OpenCV development files:

```bash
cmake -S tools/gmsl_latency_benchmark -B build/gmsl_latency_benchmark
cmake --build build/gmsl_latency_benchmark -j
```

If the OpenCV build exposes CUDA `cudawarping`, the benchmark will use
`cv::cuda::remap` by default. Add `--cpu` to force the CPU path.

## Two-Calibrated-Camera Baseline

```bash
mkdir -p artifacts/gmsl_latency

build/gmsl_latency_benchmark/gmsl_latency_benchmark \
  --camera video6=/dev/video6 \
  --camera video7=/dev/video7 \
  --frames 300 \
  --warmup 30 \
  --output-json artifacts/gmsl_latency/opencv_remap_video6_video7.json
```

The JSON report contains per-camera `read_ms`, `color_ms`, `upload_ms`,
`remap_ms`, `download_ms`, `rotate_ms`, `process_ms`, and `frame_ms` summaries.
Each summary reports `mean`, `p50`, `p95`, `p99`, and `max`.

Summarize one or more JSON reports after the run:

```bash
python3 tools/gmsl_latency_benchmark/summarize_latency_json.py \
  artifacts/gmsl_latency/capture_only_four_camera.json \
  artifacts/gmsl_latency/opencv_remap_video6_video7.json \
  --output-markdown artifacts/gmsl_latency/summary.md \
  --output-csv artifacts/gmsl_latency/summary.csv \
  --process-p95-budget-ms 4.0
```

## Capture-Only Four-Camera Baseline

Use this first when not all four cameras have intrinsics in the manifest yet:

```bash
mkdir -p artifacts/gmsl_latency

build/gmsl_latency_benchmark/gmsl_latency_benchmark \
  --capture-only \
  --raw-camera video0=/dev/video0 \
  --raw-camera video1=/dev/video1 \
  --raw-camera video6=/dev/video6 \
  --raw-camera video7=/dev/video7 \
  --frames 300 \
  --warmup 30 \
  --output-json artifacts/gmsl_latency/capture_only_four_camera.json
```

`--raw-camera` is intentionally limited to `--capture-only`. Full remap timing
requires calibrated `--camera KEY=/dev/videoN` entries that exist in
`configs/camera_intrinsics/gmsl_h190ta/manifest.json`.

## Notes

- Use `--buffer-count 1` for low-latency baseline testing; it is the default.
- Use `--width` and `--height` together only when intentionally testing another
  sensor mode or V4L2 scale mode.
- The GPU path synchronizes after upload, remap, and download so per-stage
  timings are real wall times, not merely queued CUDA work.
- This OpenCV benchmark still includes CPU capture and CPU/GPU transfer costs.
  It is a baseline, not the final 4 ms production architecture.
