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

Only `video6` and `video7` have imported intrinsics at this point. The
`pending_h190ta_*` entries are placeholders for the two same-model cameras
whose vendor intrinsics still need to be imported. Do not use those pending
entries for runtime remap.

Current confirmed physical mapping:

- `stick_top` -> `H190TA-I06031460` -> `video7`
- `stick_bottom` -> `H190TA-I06031459` -> `video6`
- `eye_left` -> pending intrinsics import
- `eye_right` -> pending intrinsics import
