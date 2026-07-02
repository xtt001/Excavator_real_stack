# GMSL Realtime Capture

Phase-1 four-camera capture core for the H190TA GMSL camera stack. This tool is
focused on direct V4L2 capture and metadata accounting:
frame sequence, V4L2 timestamp, host arrival time, flags, byte count, drop
count, and timestamp-based cross-camera skew.

The production path must use `--memory dmabuf`: NVIDIA SurfaceArray buffers are
allocated first, their DMABUF fds are queued to V4L2 with
`V4L2_MEMORY_DMABUF`, and the same fds can be imported by EGL/CUDA. `--memory
mmap` remains available only as a diagnosis fallback.

It does not encode, display, or write HDF5. The runtime path keeps only a small
per-camera latest ring, so policy-side consumers can select the latest frame or
the nearest frame to a target timestamp without accumulating old frames.

## Build

Build on the Jetson:

```bash
cmake -S tools/gmsl_realtime_capture -B build/gmsl_realtime_capture
cmake --build build/gmsl_realtime_capture -j
```

When CUDA and Jetson multimedia headers are available, the build also produces
`gmsl_realtime_preprocess_probe`.

## Four-Camera Probe

Bring up the current field devices first:

```bash
env GMSL_VIDEO_DEVICES="4 5 6 7" bash scripts/bring_up_gmsl_cameras.sh
```

Run a short capture:

```bash
mkdir -p artifacts/gmsl_realtime_capture

build/gmsl_realtime_capture/gmsl_realtime_capture \
  --camera video4=/dev/video4 \
  --camera video5=/dev/video5 \
  --camera video6=/dev/video6 \
  --camera video7=/dev/video7 \
  --memory dmabuf \
  --cuda-import-probe \
  --frames 300 \
  --warmup 8 \
  --buffers 4 \
  --ring-size 8 \
  --output-json artifacts/gmsl_realtime_capture/four_camera_300.json
```

Run the phase-1 acceptance length:

```bash
build/gmsl_realtime_capture/gmsl_realtime_capture \
  --camera video4=/dev/video4 \
  --camera video5=/dev/video5 \
  --camera video6=/dev/video6 \
  --camera video7=/dev/video7 \
  --memory dmabuf \
  --cuda-import-probe \
  --frames 3000 \
  --warmup 8 \
  --buffers 4 \
  --ring-size 8 \
  --output-json artifacts/gmsl_realtime_capture/four_camera_3000.json
```

## Output

The JSON report includes:

- `cameras.{camera}.frames_reported`
- `cameras.{camera}.timestamp_interval_ms`
- `cameras.{camera}.host_arrival_interval_ms`
- `cameras.{camera}.drop_count`
- `cameras.{camera}.error_flag_count`
- `cameras.{camera}.bytes_mismatch_count`
- `cameras.{camera}.frames[].sequence`
- `cameras.{camera}.frames[].v4l2_timestamp_ns`
- `cameras.{camera}.frames[].flags`
- `cameras.{camera}.frames[].host_arrival_mono_ns`
- `cameras.{camera}.frames[].dmabuf_fd`
- `cameras.{camera}.frames[].cuda_imported`
- `cameras.{camera}.frames[].cuda_device_ptr`
- `cameras.{camera}.frames[].cuda_pitch`
- `cameras.{camera}.frames[].cuda_frame_type`
- `cameras.{camera}.frames[].inter_frame_delta_ms`
- `sync.skew_ms`

The current timestamp contract is still the V4L2 monotonic EOF timestamp. The
tool does not assume that `V4L2_BUF_FLAG_ERROR` is harmless: it counts flagged
frames, keeps the raw flags in per-frame metadata, and separately reports byte
count, sequence, timestamp interval, and cross-camera skew.

## Notes

- In `--memory dmabuf`, `CameraFrame::data` is null and `CameraFrame::dmabuf_fd`
  is the image handle. Downstream code must import the fd/EGL/CUDA frame and
  must not read the image through a CPU pointer.
- `--cuda-import-probe` validates the zero-copy path by mapping the DMABUF to an
  EGLImage and registering it with CUDA. It records CUDA pitch-frame metadata
  without launching a preprocessing kernel.
- `--memory mmap` keeps `CameraFrame::data` for debugging only. It is not the
  production path for stage-2 preprocessing.
- For long recordings, use `--no-frame-details` or reduce `--detail-frames` to
  avoid writing very large diagnostic JSON files.

## Stage-2 Zero-Copy Preprocess Probe

Run the fused UYVY preprocess path without CPU frame copies:

```bash
build/gmsl_realtime_capture/gmsl_realtime_preprocess_probe \
  --preprocess-manifest configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json \
  --manifest configs/camera_intrinsics/gmsl_h190ta/manifest.json \
  --frames 300 \
  --warmup 8 \
  --buffers 8 \
  --output-json artifacts/gmsl_realtime_capture/four_camera_preprocess_300.json
```

The probe uses:

- `V4L2_MEMORY_DMABUF` capture buffers allocated by `NvBufSurfaceAllocate`.
- EGL/CUDA import of each dequeued DMABUF pitch frame.
- A fused CUDA kernel reading UYVY directly and writing `NHWC float32 RGB`.
- No UYVY-to-BGR/RGBA conversion and no output image download to CPU.

The stage-2 probe returns a CUDA event release fence from the frame callback.
The capture thread requeues each V4L2 buffer only after that event is complete,
so the DMABUF is not reused while the kernel is reading it and the callback does
not block on every frame.
