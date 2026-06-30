# GMSL 四路 H190TA 相机内参、外参和 110 度预处理工序

日期：2026-06-30

目标：把四路 H190TA GMSL 相机的内参、安装外参、图像预处理策略和现场证据版本化。当前已经导入
`video6` / `video7` 两路内参；另外两路同型号相机仍需要导入各自厂商内参，不能复用已导入相机的
`K/D` 数值。

## 1. 当前预处理策略

四路都先采用同一类训练/推理派生图：

- 投影：`virtual_rectilinear`
- HFOV：`110 deg`
- 输出尺寸：`384x216`
- 色彩：`RGB`
- 内容取舍：允许损失部分 190 度边缘内容，优先保证合理局部无畸变图像

当前已定：

- `stick_top`：`video7` / `H190TA-I06031460`，`pitch_down_deg = 0`，并保留 `rotate_180`
- `stick_bottom`：`video6` / `H190TA-I06031459`，`pitch_down_deg = 20`
- `eye_left`：待导入内参和现场 `/dev/videoN` 映射
- `eye_right`：待导入内参和现场 `/dev/videoN` 映射

配置入口：

```text
configs/camera_calibration/gmsl_h190ta_four_camera/camera_mount_mapping.json
configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json
```

`camera_mount_mapping.json` 是物理安装位置到 serial、camera key、device hint 和内参文件的主映射。
训练/推理的相机顺序应优先按物理位置读：

```text
stick_top, stick_bottom, eye_left, eye_right
```

## 2. 打印标定板

打印这个 SVG：

```text
docs/assets/gmsl_chessboard_8x6_25mm_a4.svg
```

打印要求：

- A4 横向。
- 实际大小 / 100% 打印。
- 不要缩放，不要使用“适合页面”。
- 打印后用尺量底部 `100 mm print scale check`，误差应小于 1 mm。
- 棋盘格参数为 `8x6` 内角点、`25 mm` 方格。

如果现场空间允许，建议把同一图案贴在硬板上，避免纸张弯曲。外参质量主要受板面平整度、尺寸误差和角点覆盖范围影响。

## 3. 导入另外两路内参

后两路拿到厂商 `H190TA-*.txt` 后，用工具导入，不要手改 JSON：

```bash
python3 tools/gmsl_camera_config/import_h190ta_intrinsics.py \
  --manifest configs/camera_intrinsics/gmsl_h190ta/manifest.json \
  --intrinsics-file /path/to/H190TA-I06031461.txt \
  --camera video0 \
  --device /dev/video0 \
  --serial H190TA-I06031461 \
  --orientation normal
```

再导入第四路：

```bash
python3 tools/gmsl_camera_config/import_h190ta_intrinsics.py \
  --manifest configs/camera_intrinsics/gmsl_h190ta/manifest.json \
  --intrinsics-file /path/to/H190TA-I06031462.txt \
  --camera video1 \
  --device /dev/video1 \
  --serial H190TA-I06031462 \
  --orientation normal
```

`video0` / `video1` 只是示例，必须按现场实际 `/dev/videoN` 映射填写。若某路画面倒置，导入时用：

```bash
--orientation rotate_180
```

导入后运行：

```bash
python3 -m json.tool configs/camera_intrinsics/gmsl_h190ta/manifest.json >/tmp/gmsl_h190ta_manifest.check.json
PYTHONPATH=$PWD/testbed python -m pytest testbed/tests/test_gmsl_camera_intrinsics.py
```

注意：同型号只表示可以复用 110 度 HFOV virtual view 策略；每个镜头仍必须使用自己的内参。

## 4. 现场采集外参证据

每路至少采集 12 张有效棋盘格图，建议 20 张。每路要覆盖：

- 近距离、远距离。
- 左上、右上、左下、右下。
- 轻微俯仰、偏航、滚转。
- 标定板占画面中部和边缘的情况。

保存路径建议：

```text
artifacts/gmsl_extrinsics_YYYYMMDD/
  video6/
  video7/
  video0/
  video1/
  mount_photos/
  notes.md
```

`notes.md` 至少记录：

- 每路相机物理安装位置和朝向。
- 每路 `/dev/videoN`、serial、camera key、mount position。
- 标定板打印尺寸复核结果。
- 标定板相对车体或安装 link 的测量方式。
- 哪些帧用于标定，哪些帧因模糊、遮挡、棋盘格不完整被剔除。

## 5. 外参求解口径

单相机看到棋盘格，只能直接得到该相机相对棋盘格的 pose。要得到可复现的
`mount_T_camera` 或 `body_T_camera`，还需要以下两种信息之一：

1. 标定板在车体或某个安装 link 坐标系中的 pose。
2. 多路相机同时看到同一块标定板，并通过共同 board frame 建立相对外参。

因此现场不要只保存“棋盘格图片”，还要保存板的位置测量、安装照片和采集说明。否则后续只能得到
camera-to-board 的临时 pose，不能得到稳定的车体外参。

外参模板：

```text
configs/camera_calibration/gmsl_h190ta_four_camera/extrinsics_template.json
```

字段约定：

- `mount_T_camera`：安装 link 坐标系到相机坐标系的变换。
- 相机坐标系采用 OpenCV 约定：`x right, y down, z forward`。
- 单位为米。
- `solve_pnp_rms_px` 必须记录重投影误差。

## 6. 通过/失败标准

通过：

- 每路内参都已导入 manifest。
- 每路相机 key、serial、`/dev/videoN`、orientation 都和现场证据一致。
- 每路至少 12 张有效棋盘格图。
- 标定板打印 100 mm 校验误差小于 1 mm。
- 外参结果能追溯到图像、板位姿记录和安装照片。

失败或需要重采：

- 打印被缩放。
- 纸张弯曲明显。
- 棋盘格只出现在画面中心，缺少边缘和倾斜姿态。
- 某路相机内参缺失却尝试做完整 remap 或外参。
- 没有记录标定板相对车体或安装 link 的位置。

## 7. 后续代码工作

这一步只准备现场采集和配置落点。拿到四路内参、标定图和板位姿记录后，再做：

1. 把 `pending_h190ta_*` 替换成正式 `videoN` camera key。
2. 更新 `camera_mount_mapping.json` 中 `eye_left` / `eye_right` 的 serial、device hint 和内参文件。
3. 用标定图求解每路 `mount_T_camera`。
4. 生成带重投影误差和 evidence 路径的正式外参 manifest。
5. 把 110 度 virtual rectilinear map 预计算到实时 CUDA/GStreamer preprocessing 路径。
