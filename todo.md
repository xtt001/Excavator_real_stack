# GMSL Four-Camera Field Todo

This todo is for the next field round after the camera mounting structure is
ready.

## 0. Current Fixed Assumptions

- Physical camera order: `stick_top`, `stick_bottom`, `eye_left`, `eye_right`.
- `stick_top` -> `H190TA-I06031460` -> `video7` -> `H190TA-I06031460.txt`.
- `stick_bottom` -> `H190TA-I06031459` -> `video6` -> `H190TA-I06031459.txt`.
- `eye_left` and `eye_right` are pending vendor intrinsics and final
  `/dev/videoN` mapping.
- Four H190TA cameras should use the same first-pass preprocessing family:
  `virtual_rectilinear`, `HFOV=110 deg`, output `384x216`.
- Current pitch choices:
  - `stick_top`: `pitch_down_deg=0`, keep `rotate_180`.
  - `stick_bottom`: `pitch_down_deg=20`.

## 1. Before Going To Site

- [ ] Print `docs/assets/gmsl_chessboard_8x6_25mm_a4.svg` on A4 landscape.
- [ ] Print at actual size / 100%; do not use fit-to-page.
- [ ] Measure the printed `100 mm` scale check line and record the measured
      value.
- [ ] Prepare a flat board backing for the print so the calibration target does
      not bend.
- [ ] Put the latest branch on the Jetson-side repo:
      `fs/online-qc-dev`.

## 2. After Mount Structure Is Installed

- [ ] Photograph each physical camera mount:
      `stick_top`, `stick_bottom`, `eye_left`, `eye_right`.
- [ ] Record cable label, serial number, and observed `/dev/videoN` for each
      physical position.
- [ ] Confirm the two known cameras still match:
  - [ ] `H190TA-I06031460` is `stick_top`.
  - [ ] `H190TA-I06031459` is `stick_bottom`.
- [ ] Update `configs/camera_calibration/gmsl_h190ta_four_camera/camera_mount_mapping.json`
      if any `/dev/videoN` mapping changes.

## 3. Import Missing Intrinsics

- [ ] Obtain vendor intrinsics txt files for `eye_left` and `eye_right`.
- [ ] Import `eye_left` with `tools/gmsl_camera_config/import_h190ta_intrinsics.py`.
- [ ] Import `eye_right` with `tools/gmsl_camera_config/import_h190ta_intrinsics.py`.
- [ ] Update `camera_mount_mapping.json` for:
  - [ ] `eye_left` serial.
  - [ ] `eye_left` camera key.
  - [ ] `eye_left` `/dev/videoN`.
  - [ ] `eye_left` intrinsics file.
  - [ ] `eye_right` serial.
  - [ ] `eye_right` camera key.
  - [ ] `eye_right` `/dev/videoN`.
  - [ ] `eye_right` intrinsics file.
- [ ] Run:

```bash
python3 -m json.tool configs/camera_intrinsics/gmsl_h190ta/manifest.json >/tmp/gmsl_intrinsics.check.json
PYTHONPATH=$PWD/testbed python -m pytest testbed/tests/test_gmsl_camera_intrinsics.py
```

## 4. Bring Up And Capture Evidence

- [ ] Run a config-only bring-up check with the real four device ids:

```bash
GMSL_PRINT_CONFIG_ONLY=1 GMSL_VIDEO_DEVICES="..." scripts/bring_up_gmsl_cameras.sh
```

- [ ] Run real GMSL bring-up with the same four device ids.
- [ ] Save `v4l2-ctl --all` output for each camera.
- [ ] Capture raw frame evidence for all four cameras.
- [ ] Save a four-camera contact sheet.
- [ ] Confirm and record orientation for each camera.
- [ ] Check whether `eye_left` / `eye_right` need `rotate_180`.
- [ ] Check whether `eye_left` / `eye_right` need a non-zero
      `pitch_down_deg`.

## 5. Extrinsic Calibration Evidence

- [ ] Capture at least 12 valid checkerboard frames per camera; target 20.
- [ ] Cover near, far, left, right, upper, lower, and tilted board poses.
- [ ] Save rejected frames separately or list why they were rejected.
- [ ] Record the checkerboard pose relative to the machine body or the relevant
      mount link when possible.
- [ ] Save all calibration images under a dated directory, for example:

```text
artifacts/gmsl_extrinsics_YYYYMMDD/
  stick_top/
  stick_bottom/
  eye_left/
  eye_right/
  mount_photos/
  notes.md
```

- [ ] Fill `notes.md` with serial, `/dev/videoN`, mount position, target size
      check, and board pose measurement notes.
- [ ] Use the evidence to replace
      `configs/camera_calibration/gmsl_h190ta_four_camera/extrinsics_template.json`
      with a measured extrinsics manifest.

## 6. Latency Benchmark

- [ ] Build the latency benchmark on the Jetson:

```bash
cmake -S tools/gmsl_latency_benchmark -B build/gmsl_latency_benchmark
cmake --build build/gmsl_latency_benchmark -j
```

- [ ] Run capture-only with all four cameras.
- [ ] Run OpenCV remap benchmark for the calibrated cameras.
- [ ] After all four intrinsics are imported, run four-camera remap benchmark.
- [ ] Save JSON results under `artifacts/gmsl_latency/`.
- [ ] Record `tegrastats` during each benchmark.
- [ ] Decide whether OpenCV path is enough or whether we must move to
      GStreamer/NVMM plus fused CUDA preprocessing.

## 7. Preprocessing And Training Readiness

- [ ] Update `preprocess_manifest.json` after final orientation and pitch are
      confirmed for all four positions.
- [ ] Generate visual samples for all four 110 deg virtual rectilinear views.
- [ ] Verify train/inference camera order remains:
      `stick_top`, `stick_bottom`, `eye_left`, `eye_right`.
- [ ] Decide first training input set:
  - [ ] two stick cameras only.
  - [ ] four cameras.
  - [ ] ablation against existing FPV baseline.
- [ ] Do not train with pending intrinsics, pending orientation, or unverified
      transform versions.

## 8. Completion Criteria For This Field Round

- [ ] Four cameras have stable `/dev/videoN` mapping.
- [ ] Four camera serials are mapped to physical mount positions.
- [ ] Four intrinsics are imported and validated.
- [ ] Four orientations are confirmed.
- [ ] Four raw evidence frames and one contact sheet are saved.
- [ ] Four-camera latency benchmark JSON exists.
- [ ] Calibration target images and mount photos are saved.
- [ ] Next code step is clear: measured extrinsics manifest or realtime CUDA
      preprocessing implementation.
