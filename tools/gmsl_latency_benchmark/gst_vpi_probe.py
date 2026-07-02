#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraArg:
    key: str
    device: str


@dataclass(frozen=True)
class CameraConfig:
    key: str
    device: str
    serial: str
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    k: np.ndarray
    d: np.ndarray
    hfov_deg: float
    yaw_deg: float
    pitch_down_deg: float
    roll_deg: float


class GstRunner:
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        io_mode: str,
        gst_format: str,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.io_mode = io_mode
        self.gst_format = gst_format
        self.pipeline_desc = build_pipeline(device, width, height, fps, io_mode, gst_format)

        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        self._gst = Gst
        self.pipeline = Gst.parse_launch(self.pipeline_desc)
        self.sink = self.pipeline.get_by_name("sink")
        if self.sink is None:
            raise RuntimeError("appsink named 'sink' was not created")

    def start(self) -> None:
        ret = self.pipeline.set_state(self._gst.State.PLAYING)
        if ret == self._gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"failed to start pipeline: {self.pipeline_desc}")
        state_ret, _, _ = self.pipeline.get_state(5 * self._gst.SECOND)
        if state_ret == self._gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"pipeline did not reach PLAYING: {self.pipeline_desc}")

    def stop(self) -> None:
        self.pipeline.set_state(self._gst.State.NULL)

    def pull_sample(self, timeout_sec: float):
        timeout_ns = int(timeout_sec * self._gst.SECOND)
        return self.sink.emit("try-pull-sample", timeout_ns)

    def bus_error(self) -> str:
        msg = self.pipeline.get_bus().pop_filtered(
            self._gst.MessageType.ERROR | self._gst.MessageType.EOS
        )
        if msg is None:
            return ""
        if msg.type == self._gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            return f"{err.message}; debug={debug}"
        return "EOS"


def build_pipeline(
    device: str,
    width: int,
    height: int,
    fps: int,
    io_mode: str,
    gst_format: str,
) -> str:
    source = (
        f"v4l2src device={device} do-timestamp=true io-mode={io_mode} ! "
        f"video/x-raw,format=UYVY,width={width},height={height},framerate={fps}/1"
    )
    if gst_format == "uyvy":
        body = source
    elif gst_format == "rgba":
        body = (
            f"{source} ! nvvidconv ! "
            f"video/x-raw,format=RGBA,width={width},height={height},framerate={fps}/1"
        )
    else:
        raise ValueError(f"unsupported gst_format: {gst_format}")
    return (
        f"{body} ! appsink name=sink emit-signals=false sync=false "
        "max-buffers=1 drop=true"
    )


def parse_camera(raw: str) -> CameraArg:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("camera must be KEY=/dev/videoN")
    key, device = raw.split("=", 1)
    if not key or not device:
        raise argparse.ArgumentTypeError("camera must be KEY=/dev/videoN")
    return CameraArg(key=key, device=device)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(np.max(arr)),
    }


def is_valid_gst_time(value: int) -> bool:
    from gi.repository import Gst

    return value != Gst.CLOCK_TIME_NONE


def _deg_to_rad(degrees: float) -> float:
    return degrees * math.pi / 180.0


def _rotation_x(degrees: float) -> np.ndarray:
    t = _deg_to_rad(degrees)
    c = math.cos(t)
    s = math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(degrees: float) -> np.ndarray:
    t = _deg_to_rad(degrees)
    c = math.cos(t)
    s = math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_z(degrees: float) -> np.ndarray:
    t = _deg_to_rad(degrees)
    c = math.cos(t)
    s = math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_virtual_rectilinear_map(cfg: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(cfg.output_width, dtype=np.float64)
    y = np.arange(cfg.output_height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x, y)

    focal = (cfg.output_width * 0.5) / math.tan(_deg_to_rad(cfg.hfov_deg) * 0.5)
    cx_out = (cfg.output_width - 1.0) * 0.5
    cy_out = (cfg.output_height - 1.0) * 0.5

    vx = (grid_x - cx_out) / focal
    vy = (grid_y - cy_out) / focal
    vz = np.ones_like(vx)

    rot = _rotation_z(cfg.roll_deg) @ _rotation_y(cfg.yaw_deg) @ _rotation_x(-cfg.pitch_down_deg)
    sx = rot[0, 0] * vx + rot[0, 1] * vy + rot[0, 2] * vz
    sy = rot[1, 0] * vx + rot[1, 1] * vy + rot[1, 2] * vz
    sz = rot[2, 0] * vx + rot[2, 1] * vy + rot[2, 2] * vz
    xn = sx / sz
    yn = sy / sz
    radius = np.sqrt(xn * xn + yn * yn)

    theta = np.arctan(radius)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    k1, k2, k3, k4 = cfg.d.tolist()
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    scale = np.ones_like(radius)
    np.divide(theta_d, radius, out=scale, where=radius > 1e-12)

    map_x = cfg.k[0, 0] * xn * scale + cfg.k[0, 2]
    map_y = cfg.k[1, 1] * yn * scale + cfg.k[1, 2]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def build_vpi_warp_map(cfg: CameraConfig):
    import vpi

    map_x, map_y = build_virtual_rectilinear_map(cfg)
    warp_map = vpi.WarpMap((cfg.output_width, cfg.output_height), interval=1)
    control = np.asarray(warp_map)
    ys = np.minimum(np.arange(control.shape[0]), cfg.output_height - 1)
    xs = np.minimum(np.arange(control.shape[1]), cfg.output_width - 1)
    control[..., 0] = map_x[np.ix_(ys, xs)]
    control[..., 1] = map_y[np.ix_(ys, xs)]
    return warp_map


def load_camera_configs(
    cameras: list[CameraArg],
    intrinsics_manifest_path: Path,
    preprocess_manifest_path: Path,
) -> dict[str, CameraConfig]:
    intrinsics = json.loads(intrinsics_manifest_path.read_text(encoding="utf-8"))
    preprocess = json.loads(preprocess_manifest_path.read_text(encoding="utf-8"))
    output = preprocess["output"]
    output_width = int(output["width"])
    output_height = int(output["height"])

    intr_by_key = {cam["camera_key"]: cam for cam in intrinsics["cameras"]}
    prep_by_key = {cam["camera_key"]: cam for cam in preprocess["cameras"]}

    configs: dict[str, CameraConfig] = {}
    for camera in cameras:
        if camera.key not in intr_by_key:
            raise KeyError(f"{camera.key} missing from {intrinsics_manifest_path}")
        if camera.key not in prep_by_key:
            raise KeyError(f"{camera.key} missing from {preprocess_manifest_path}")

        intr = intr_by_key[camera.key]
        transform = prep_by_key[camera.key]["transform"]
        if transform["projection"] != "virtual_rectilinear":
            raise ValueError(f"unsupported projection for {camera.key}: {transform['projection']}")

        configs[camera.key] = CameraConfig(
            key=camera.key,
            device=camera.device,
            serial=intr["serial"],
            input_width=int(intrinsics["image_width"]),
            input_height=int(intrinsics["image_height"]),
            output_width=output_width,
            output_height=output_height,
            k=np.asarray(intr["K"], dtype=np.float64),
            d=np.asarray(intr["D"], dtype=np.float64),
            hfov_deg=float(transform["hfov_deg"]),
            yaw_deg=float(transform.get("yaw_deg", 0.0)),
            pitch_down_deg=float(transform.get("pitch_down_deg", 0.0)),
            roll_deg=float(transform.get("roll_deg", 0.0)),
        )
    return configs


def timed_buffer_map(sample, expected_size: int) -> tuple[Any, Any, float]:
    from gi.repository import Gst

    buf = sample.get_buffer()
    start = time.perf_counter_ns()
    ok, map_info = buf.map(Gst.MapFlags.READ)
    end = time.perf_counter_ns()
    if not ok:
        raise RuntimeError("failed to map Gst buffer")
    if map_info.size < expected_size:
        buf.unmap(map_info)
        raise RuntimeError(f"Gst buffer too small: got {map_info.size}, expected {expected_size}")
    return buf, map_info, (end - start) / 1e6


def run_capture_camera(args: argparse.Namespace, camera: CameraArg) -> dict[str, Any]:
    runner = GstRunner(
        device=camera.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        io_mode=args.io_mode,
        gst_format=args.gst_format,
    )
    metrics: dict[str, list[float]] = {
        "pull_ms": [],
        "map_ms": [],
        "frame_ms": [],
        "host_interval_ms": [],
        "pts_interval_ms": [],
        "duration_ms": [],
    }
    timeouts = 0
    frames_seen = 0
    measured_frames = 0
    prev_arrival_ns: int | None = None
    bytes_per_pixel = 2 if args.gst_format == "uyvy" else 4
    expected_size = args.width * args.height * bytes_per_pixel
    checksum = 0
    prev_pts_ns: int | None = None
    first_pts_ns: int | None = None
    last_pts_ns: int | None = None

    try:
        runner.start()
        for i in range(args.warmup + args.frames):
            frame_start = time.perf_counter_ns()
            pull_start = time.perf_counter_ns()
            sample = runner.pull_sample(args.timeout_sec)
            arrival_ns = time.perf_counter_ns()
            if sample is None:
                timeouts += 1
                bus_error = runner.bus_error()
                if bus_error:
                    raise RuntimeError(bus_error)
                continue

            buf, map_info, map_ms = timed_buffer_map(sample, expected_size)
            pts_ns = int(buf.pts)
            duration_ns = int(buf.duration)
            valid_pts = is_valid_gst_time(pts_ns)
            valid_duration = is_valid_gst_time(duration_ns)
            data = memoryview(map_info.data)
            if len(data) >= expected_size:
                checksum ^= int(data[0])
                checksum ^= int(data[expected_size - 1])
            buf.unmap(map_info)

            frames_seen += 1
            if i >= args.warmup:
                measured_frames += 1
                metrics["pull_ms"].append((arrival_ns - pull_start) / 1e6)
                metrics["map_ms"].append(map_ms)
                metrics["frame_ms"].append((time.perf_counter_ns() - frame_start) / 1e6)
                if prev_arrival_ns is not None:
                    metrics["host_interval_ms"].append((arrival_ns - prev_arrival_ns) / 1e6)
                if valid_pts:
                    if first_pts_ns is None:
                        first_pts_ns = pts_ns
                    last_pts_ns = pts_ns
                    if prev_pts_ns is not None:
                        metrics["pts_interval_ms"].append((pts_ns - prev_pts_ns) / 1e6)
                    prev_pts_ns = pts_ns
                if valid_duration:
                    metrics["duration_ms"].append(duration_ns / 1e6)
                prev_arrival_ns = arrival_ns
            else:
                prev_arrival_ns = None
                prev_pts_ns = None
    finally:
        runner.stop()

    return {
        "camera_key": camera.key,
        "device": camera.device,
        "pipeline": runner.pipeline_desc,
        "frames_seen": frames_seen,
        "measured_frames": measured_frames,
        "timeouts": timeouts,
        "checksum": checksum,
        "first_pts_ns": first_pts_ns,
        "last_pts_ns": last_pts_ns,
        "metrics": {name: summarize(values) for name, values in metrics.items()},
    }


def run_vpi_camera(args: argparse.Namespace, cfg: CameraConfig) -> dict[str, Any]:
    import vpi

    runner = GstRunner(
        device=cfg.device,
        width=cfg.input_width,
        height=cfg.input_height,
        fps=args.fps,
        io_mode=args.io_mode,
        gst_format="rgba",
    )
    warp_map = build_vpi_warp_map(cfg)
    stream = vpi.Stream()
    rect_rgba = vpi.Image((cfg.output_width, cfg.output_height), vpi.Format.RGBA8)
    rect_rgb = vpi.Image((cfg.output_width, cfg.output_height), vpi.Format.RGB8)

    metrics: dict[str, list[float]] = {
        "pull_ms": [],
        "map_ms": [],
        "wrap_ms": [],
        "remap_ms": [],
        "rgb_ms": [],
        "download_ms": [],
        "process_ms": [],
        "process_no_download_ms": [],
        "frame_ms": [],
        "host_interval_ms": [],
        "pts_interval_ms": [],
        "duration_ms": [],
    }
    timeouts = 0
    frames_seen = 0
    measured_frames = 0
    prev_arrival_ns: int | None = None
    expected_size = cfg.input_width * cfg.input_height * 4
    checksum = 0
    prev_pts_ns: int | None = None
    first_pts_ns: int | None = None
    last_pts_ns: int | None = None

    try:
        runner.start()
        for i in range(args.warmup + args.frames):
            measured = i >= args.warmup
            frame_start = time.perf_counter_ns()
            pull_start = time.perf_counter_ns()
            sample = runner.pull_sample(args.timeout_sec)
            arrival_ns = time.perf_counter_ns()
            if sample is None:
                timeouts += 1
                bus_error = runner.bus_error()
                if bus_error:
                    raise RuntimeError(bus_error)
                continue

            process_start = time.perf_counter_ns()
            buf, map_info, map_ms = timed_buffer_map(sample, expected_size)
            pts_ns = int(buf.pts)
            duration_ns = int(buf.duration)
            valid_pts = is_valid_gst_time(pts_ns)
            valid_duration = is_valid_gst_time(duration_ns)
            try:
                image_array = np.ndarray(
                    shape=(cfg.input_height, cfg.input_width, 4),
                    dtype=np.uint8,
                    buffer=map_info.data,
                )

                wrap_start = time.perf_counter_ns()
                image = vpi.asimage(image_array, vpi.Format.RGBA8)
                wrap_end = time.perf_counter_ns()

                remap_start = time.perf_counter_ns()
                with stream, vpi.Backend.CUDA:
                    image.remap(
                        warp_map,
                        size=(cfg.output_width, cfg.output_height),
                        out=rect_rgba,
                        interp=vpi.Interp.LINEAR,
                        border=vpi.Border.ZERO,
                    )
                stream.sync()
                remap_end = time.perf_counter_ns()

                rgb_ms = 0.0
                output_image = rect_rgba
                if args.rgb_output:
                    rgb_start = time.perf_counter_ns()
                    with stream, vpi.Backend.CUDA:
                        rect_rgba.convert(rect_rgb)
                    stream.sync()
                    rgb_end = time.perf_counter_ns()
                    rgb_ms = (rgb_end - rgb_start) / 1e6
                    output_image = rect_rgb

                download_ms = 0.0
                if args.download:
                    download_start = time.perf_counter_ns()
                    out_cpu = output_image.cpu()
                    download_end = time.perf_counter_ns()
                    download_ms = (download_end - download_start) / 1e6
                    checksum ^= int(out_cpu.reshape(-1)[0])

                process_end = time.perf_counter_ns()
            finally:
                buf.unmap(map_info)

            frames_seen += 1
            if measured:
                measured_frames += 1
                pull_ms = (arrival_ns - pull_start) / 1e6
                wrap_ms = (wrap_end - wrap_start) / 1e6
                remap_ms = (remap_end - remap_start) / 1e6
                process_no_download_ms = (
                    map_ms + wrap_ms + remap_ms + rgb_ms
                )
                process_ms = (process_end - process_start) / 1e6
                metrics["pull_ms"].append(pull_ms)
                metrics["map_ms"].append(map_ms)
                metrics["wrap_ms"].append(wrap_ms)
                metrics["remap_ms"].append(remap_ms)
                metrics["rgb_ms"].append(rgb_ms)
                metrics["download_ms"].append(download_ms)
                metrics["process_no_download_ms"].append(process_no_download_ms)
                metrics["process_ms"].append(process_ms)
                metrics["frame_ms"].append((time.perf_counter_ns() - frame_start) / 1e6)
                if prev_arrival_ns is not None:
                    metrics["host_interval_ms"].append((arrival_ns - prev_arrival_ns) / 1e6)
                if valid_pts:
                    if first_pts_ns is None:
                        first_pts_ns = pts_ns
                    last_pts_ns = pts_ns
                    if prev_pts_ns is not None:
                        metrics["pts_interval_ms"].append((pts_ns - prev_pts_ns) / 1e6)
                    prev_pts_ns = pts_ns
                if valid_duration:
                    metrics["duration_ms"].append(duration_ns / 1e6)
                prev_arrival_ns = arrival_ns
            else:
                prev_arrival_ns = None
                prev_pts_ns = None
    finally:
        runner.stop()

    return {
        "camera_key": cfg.key,
        "device": cfg.device,
        "serial": cfg.serial,
        "pipeline": runner.pipeline_desc,
        "input_width": cfg.input_width,
        "input_height": cfg.input_height,
        "output_width": cfg.output_width,
        "output_height": cfg.output_height,
        "hfov_deg": cfg.hfov_deg,
        "pitch_down_deg": cfg.pitch_down_deg,
        "frames_seen": frames_seen,
        "measured_frames": measured_frames,
        "timeouts": timeouts,
        "checksum": checksum,
        "first_pts_ns": first_pts_ns,
        "last_pts_ns": last_pts_ns,
        "metrics": {name: summarize(values) for name, values in metrics.items()},
    }


def run_parallel(cameras: list[Any], worker) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(cameras)
    errors: list[str | None] = [None] * len(cameras)

    def run_one(index: int, camera: Any) -> None:
        try:
            results[index] = worker(camera)
        except Exception as exc:  # pragma: no cover - field diagnostics path
            errors[index] = repr(exc)

    threads = [
        threading.Thread(target=run_one, args=(index, camera), daemon=False)
        for index, camera in enumerate(cameras)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    output: list[dict[str, Any]] = []
    for index, camera in enumerate(cameras):
        if results[index] is not None:
            output.append(results[index] or {})
        else:
            key = getattr(camera, "key", f"camera{index}")
            device = getattr(camera, "device", "")
            output.append({"camera_key": key, "device": device, "error": errors[index]})
    return output


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print(
        f"{report['mode']} source={report.get('source', 'v4l2src')} "
        f"format={report.get('gst_format', 'rgba')} frames={report['frames']}"
    )
    for camera in report["cameras"]:
        if camera.get("error"):
            print(f"  {camera['camera_key']} {camera.get('device', '')}: ERROR {camera['error']}")
            continue
        metrics = camera.get("metrics", {})
        process = metrics.get("process_ms", {})
        remap = metrics.get("remap_ms", {})
        interval = metrics.get("host_interval_ms", {})
        pull = metrics.get("pull_ms", {})
        print(
            f"  {camera['camera_key']} {camera['device']}: "
            f"frames={camera['measured_frames']} timeouts={camera['timeouts']} "
            f"pull_p95={pull.get('p95')} "
            f"process_p95={process.get('p95')} "
            f"remap_p95={remap.get('p95')} "
            f"interval_p95={interval.get('p95')}"
        )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", action="append", type=parse_camera, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--io-mode", default="dmabuf", choices=["mmap", "dmabuf", "userptr", "auto"])
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GStreamer capture and VPI CUDA remap probe for GMSL cameras."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    capture = subparsers.add_parser("gst-capture")
    add_common_args(capture)
    capture.add_argument("--width", type=int, default=1920)
    capture.add_argument("--height", type=int, default=1536)
    capture.add_argument("--gst-format", default="uyvy", choices=["uyvy", "rgba"])

    vpi_remap = subparsers.add_parser("vpi-remap")
    add_common_args(vpi_remap)
    vpi_remap.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/camera_intrinsics/gmsl_h190ta/manifest.json"),
    )
    vpi_remap.add_argument(
        "--preprocess-manifest",
        type=Path,
        default=Path("configs/camera_calibration/gmsl_h190ta_four_camera/preprocess_manifest.json"),
    )
    vpi_remap.add_argument("--rgb-output", action="store_true")
    vpi_remap.add_argument("--download", action="store_true")
    return parser


def main() -> int:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)

    parser = build_arg_parser()
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.mode == "gst-capture":
        report = {
            "version": "gmsl_gst_vpi_probe_20260630",
            "mode": args.mode,
            "source": "v4l2src",
            "gst_format": args.gst_format,
            "io_mode": args.io_mode,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frames": args.frames,
            "warmup": args.warmup,
            "started_at": started,
            "cameras": run_parallel(args.camera, lambda cam: run_capture_camera(args, cam)),
        }
    elif args.mode == "vpi-remap":
        configs = load_camera_configs(args.camera, args.manifest, args.preprocess_manifest)
        ordered_configs = [configs[camera.key] for camera in args.camera]
        report = {
            "version": "gmsl_gst_vpi_probe_20260630",
            "mode": args.mode,
            "source": "v4l2src",
            "gst_format": "rgba",
            "io_mode": args.io_mode,
            "backend": "vpi_cuda",
            "rgb_output": args.rgb_output,
            "download": args.download,
            "manifest": str(args.manifest),
            "preprocess_manifest": str(args.preprocess_manifest),
            "fps": args.fps,
            "frames": args.frames,
            "warmup": args.warmup,
            "started_at": started,
            "cameras": run_parallel(ordered_configs, lambda cam: run_vpi_camera(args, cam)),
        }
    else:
        raise AssertionError(args.mode)

    print_summary(report)
    write_report(args.output_json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
