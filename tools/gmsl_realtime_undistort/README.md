# GMSL Realtime Undistort

Standalone C++ tool for the H190TA GMSL fisheye cameras. It reads
`configs/camera_intrinsics/gmsl_h190ta/manifest.json`, precomputes OpenCV
fisheye undistortion maps once, and then remaps each frame. If OpenCV was built
with CUDA `cudawarping` and a CUDA device is available, frame remap runs through
`cv::cuda::remap`; otherwise the same command falls back to CPU OpenCV.

Build on the Jetson or another machine with OpenCV development files:

```bash
cmake -S tools/gmsl_realtime_undistort -B build/gmsl_realtime_undistort
cmake --build build/gmsl_realtime_undistort -j
```

Offline evidence frame smoke check:

```bash
build/gmsl_realtime_undistort/gmsl_realtime_undistort \
  --camera video7 \
  --input-image docs/evidence/gmsl_frames_20260629_190917/video7_frame00.jpg \
  --output-image /tmp/video7_undistorted.jpg
```

Realtime preview from the current evidence mapping:

```bash
build/gmsl_realtime_undistort/gmsl_realtime_undistort --camera video6 --display
build/gmsl_realtime_undistort/gmsl_realtime_undistort --camera video7 --display
```

`video7` is mapped to `H190TA-I06031460` and is normal orientation in the
current install. Use `--no-rotate` only when diagnosing a temporary manifest
orientation override.
