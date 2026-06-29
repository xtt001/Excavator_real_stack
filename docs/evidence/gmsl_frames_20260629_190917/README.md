# GMSL Frame Evidence 2026-06-29

Source: slave Jetson `slave-jetson`.

Driver package:
`/home/mundane/SG8A_AGON_G2Y_A1_AGX_Orin_YUV_JP6.2_L4TR36.4.3`

Capture settings:

- `/dev/video6` and `/dev/video7`
- `1920x1536`
- `UYVY`
- `sensor_mode=2`
- `trig_mode=2`
- `trig_pin=0x00020007`
- 5 frames per camera

Original raw captures were stored on the slave under:

`/media/mundane/D/gmsl_frames_20260629_190917`

This directory intentionally commits the JPEG frame exports, frame statistics,
and SHA-256 manifest, not the raw `*.uyvy` files.
