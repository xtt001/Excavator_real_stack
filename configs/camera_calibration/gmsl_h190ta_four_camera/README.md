# GMSL H190TA Four-Camera Calibration Config

This directory tracks the four-camera calibration state for H190TA GMSL
cameras.

- `preprocess_manifest.json` records the current ACT image preprocessing
  policy: 110 degree HFOV virtual rectilinear views, `384x216` RGB output, and
  per-camera pitch/orientation choices.
- `extrinsics_template.json` records the field calibration contract and
  placeholders for `mount_T_camera`.

Only `video6` and `video7` have imported intrinsics at this point. The
`pending_h190ta_*` entries are placeholders for the two same-model cameras
whose vendor intrinsics still need to be imported. Do not use those pending
entries for runtime remap.
