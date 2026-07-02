# GMSL H190TA Intrinsics

This directory stores the raw vendor calibration text files and a structured
manifest for the four H190TA fisheye cameras used by the current field setup.

Frame evidence mapping:

- `/dev/video6` -> `H190TA-I06031459`, normal orientation.
- `/dev/video7` -> `H190TA-I06031460`, normal orientation in the current install.
- `/dev/video4` -> `H190TA-I06031461`, normal orientation.
- `/dev/video5` -> `H190TA-I06031462`, normal orientation.

The distortion model is OpenCV fisheye with `K` and `D = [k1, k2, k3, k4]`.
Runtime code should keep raw captures and transformed outputs versioned so
training and live inference use the same camera order, orientation, and
undistortion settings.

Use `tools/gmsl_camera_config/import_h190ta_intrinsics.py` to import additional
same-model H190TA vendor `*.txt` intrinsics into `manifest.json`; do not copy
`K/D` values from another physical lens.
