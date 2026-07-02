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

For current ACT-style image input timing, pass
`--preprocess-manifest configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json`.
That path uses the configured `384x216` virtual rectilinear output, 110 degree
HFOV, per-camera pitch, and manifest orientation.

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

## Four-Camera Current-Preprocess Baseline

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

The output JSON records the runtime input size, policy output size, projection,
HFOV, and `pitch_down_deg` for each camera.

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
  --raw-camera video4=/dev/video4 \
  --raw-camera video5=/dev/video5 \
  --raw-camera video6=/dev/video6 \
  --raw-camera video7=/dev/video7 \
  --frames 300 \
  --warmup 30 \
  --output-json artifacts/gmsl_latency/capture_only_four_camera.json
```

`--raw-camera` is intentionally limited to `--capture-only`. Full remap timing
requires calibrated `--camera KEY=/dev/videoN` entries that exist in
`configs/camera_intrinsics/gmsl_h190ta/manifest.json`.

## GStreamer/VPI Probe

`gst_vpi_probe.py` is a field probe for alternatives to the OpenCV CPU path.
It uses GStreamer `v4l2src`/`appsink` for capture and can route RGBA frames
through VPI CUDA remap using the same current preprocessing manifest:

```bash
python3 tools/gmsl_latency_benchmark/gst_vpi_probe.py vpi-remap \
  --camera video4=/dev/video4 \
  --camera video5=/dev/video5 \
  --frames 300 \
  --warmup 30 \
  --download \
  --output-json artifacts/gmsl_latency/vpi_remap_eye_only_download.json
```

Use `gst-capture --gst-format uyvy` to measure capture without conversion, and
`gst-capture --gst-format rgba` to include `nvvidconv` conversion before
`appsink`.

## V4L2 Timestamp Probe

Use `v4l2_timestamp_probe` to inspect camera buffer timestamps through direct
V4L2 `ioctl`, without OpenCV, GStreamer, or libv4l wrapping:

```bash
build/gmsl_latency_benchmark/v4l2_timestamp_probe \
  --device /dev/video4 \
  --frames 120 \
  --warmup 8 \
  --buffers 4 \
  --output-json artifacts/gmsl_latency/video4_timestamp.json
```

Current SG8A/SG3S-ISX031 field result on 2026-07-01:

- `/dev/video4` to `/dev/video7` report `timestamp_clock=monotonic`.
- The V4L2 `timestamp_source` is `eof`, not SOF.
- Timestamp interval p50/p95 is `33.334 ms` at 30 fps.
- The SG8A device tree sets `embedded_metadata_height = 0`, so the current
  image stream does not expose an extra embedded-metadata row for a camera-side
  exposure timestamp.

Treat this timestamp as the Jetson V4L2/VI buffer timestamp for queuing and QC.
If exposure-start timestamps are required, the next step is vendor/driver work:
enable sensor embedded metadata if the camera supports it, or patch/verify the
Tegra VI driver timestamp source.

## Notes

- Use `--buffer-count 1` for low-latency baseline testing; it is the default.
- Use `--width` and `--height` together only when intentionally testing another
  sensor mode or V4L2 scale mode.
- The GPU path synchronizes after upload, remap, and download so per-stage
  timings are real wall times, not merely queued CUDA work.
- This OpenCV benchmark still includes CPU capture and CPU/GPU transfer costs.
  It is a baseline, not the final 4 ms production architecture.
