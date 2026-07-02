# GMSL H190TA Four-Camera Calibration Config

This directory tracks the four-camera calibration state for H190TA GMSL
cameras.

- `camera_mount_mapping.json` is the source-of-truth mapping from physical
  mount positions to camera keys, serials, device hints, and intrinsics files.
- `preprocess_manifest.json` records the current ACT image preprocessing
  policy: 110 degree HFOV virtual rectilinear views, `384x216` RGB output, and
  per-camera pitch/orientation choices.
- `extrinsics_template.json` records the field calibration contract and
  placeholders for `mount_T_camera`.

Useful checks and field capture helpers:

```bash
python3 tools/gmsl_camera_config/validate_gmsl_camera_config.py
python3 tools/gmsl_camera_config/capture_gmsl_contact_sheet.py --dry-run-plan --json
```

If a future camera is marked `pending_import`, the contact-sheet helper skips it
by default. Use `--require-all` to make pending entries a hard error.

All four H190TA vendor intrinsics are imported. `video4` and `video5` use
`pitch_down_deg=10` based on field preview and still need extrinsics before
final policy runtime use.

Current confirmed physical mapping:

- `stick_top` -> `H190TA-I06031460` -> `video7`
- `stick_bottom` -> `H190TA-I06031459` -> `video6`
- `eye_left` -> `H190TA-I06031461` -> `video4` -> `pitch_down_deg=10`
- `eye_right` -> `H190TA-I06031462` -> `video5` -> `pitch_down_deg=10`
