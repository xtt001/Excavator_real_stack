"""主端 / 从端录制部署：默认 bridge 与 HDF5 路径。"""

from __future__ import annotations

import copy
import logging
import os
import socket
from typing import Any, Literal

log = logging.getLogger(__name__)

DataSide = Literal["host", "slave"]
DATA_SIDE_HOST: DataSide = "host"
DATA_SIDE_SLAVE: DataSide = "slave"
_VALID_DATA_SIDES = frozenset({DATA_SIDE_HOST, DATA_SIDE_SLAVE})

_DEFAULT_SIDE_DEFAULTS: dict[str, dict[str, Any]] = {
    DATA_SIDE_HOST: {
        "dataset_dir": "data/real_teleop_v1",
        "bridge": {"host": "127.0.0.1", "port": 8765},
    },
    DATA_SIDE_SLAVE: {
        "dataset_dir": "/data/real_teleop_v1",
        "bridge": {"host": "127.0.0.1", "port": 8765},
    },
}


def normalize_data_side(value: str | None) -> DataSide | None:
    if value is None or str(value).strip() == "":
        return None
    side = str(value).strip().lower()
    if side not in _VALID_DATA_SIDES:
        raise ValueError(
            f"data_side must be 'host' or 'slave', got {value!r}. "
            "Run tb-record-real on the machine where HDF5 should be written."
        )
    return side  # type: ignore[return-value]


def resolve_data_side(*, cfg: dict[str, Any], cli_data_side: str | None) -> DataSide | None:
    real_cfg = cfg.get("real", {})
    env_side = os.environ.get("EXCAVATOR_DATA_SIDE")
    return normalize_data_side(cli_data_side or real_cfg.get("data_side") or env_side)


def apply_data_side_config(
    cfg: dict[str, Any],
    *,
    data_side: str | None = None,
    cli_output_dir: str | None = None,
    cli_bridge_host: str | None = None,
    cli_bridge_port: int | None = None,
) -> DataSide | None:
    """
    按 host/slave 填充 dataset_dir 与 bridge 默认值（CLI 显式参数优先）。

    host：主端落盘，bridge 默认指向 EXCAVATOR_BRIDGE_HOST（从机 gateway）。
    slave：从端落盘，bridge 默认本机 127.0.0.1（须在从机运行 tb-record-real）。
    """
    side = resolve_data_side(cfg=cfg, cli_data_side=data_side)
    if side is None:
        return None

    real_cfg = cfg.setdefault("real", {})
    task_cfg = cfg.setdefault("task", {})
    bridge_cfg = real_cfg.setdefault("bridge", {})
    real_cfg["data_side"] = side

    yaml_defaults = real_cfg.get("data_side_defaults", {})
    if not isinstance(yaml_defaults, dict):
        yaml_defaults = {}
    side_defaults = copy.deepcopy(_DEFAULT_SIDE_DEFAULTS.get(side, {}))
    yaml_side = yaml_defaults.get(side, {})
    if isinstance(yaml_side, dict):
        if "dataset_dir" in yaml_side:
            side_defaults["dataset_dir"] = yaml_side["dataset_dir"]
        yaml_bridge = yaml_side.get("bridge", {})
        if isinstance(yaml_bridge, dict):
            side_defaults.setdefault("bridge", {})
            side_defaults["bridge"].update(yaml_bridge)

    env_dataset = os.environ.get(
        "EXCAVATOR_SLAVE_DATASET_DIR" if side == DATA_SIDE_SLAVE else "EXCAVATOR_HOST_DATASET_DIR"
    )
    if env_dataset:
        side_defaults["dataset_dir"] = env_dataset

    bridge_defaults = dict(side_defaults.get("bridge", {}))
    env_bridge_host = os.environ.get("EXCAVATOR_BRIDGE_HOST")
    if env_bridge_host and side == DATA_SIDE_HOST:
        bridge_defaults["host"] = env_bridge_host
    env_bridge_port = os.environ.get("EXCAVATOR_BRIDGE_PORT")
    if env_bridge_port:
        try:
            bridge_defaults["port"] = int(env_bridge_port)
        except ValueError:
            log.warning("Invalid EXCAVATOR_BRIDGE_PORT=%r, ignored.", env_bridge_port)

    if cli_output_dir is None and side_defaults.get("dataset_dir"):
        task_cfg["dataset_dir"] = str(side_defaults["dataset_dir"])

    if cli_bridge_host is None and bridge_defaults.get("host"):
        bridge_cfg["host"] = str(bridge_defaults["host"])

    if cli_bridge_port is None:
        port = int(bridge_cfg.get("port", 0) or 0)
        default_port = int(bridge_defaults.get("port", 8765))
        if port <= 0:
            bridge_cfg["port"] = default_port

    _log_data_side_hints(side, task_cfg, bridge_cfg)
    return side


def _log_data_side_hints(side: DataSide, task_cfg: dict[str, Any], bridge_cfg: dict[str, Any]) -> None:
    dataset_dir = task_cfg.get("dataset_dir", "")
    bridge_host = bridge_cfg.get("host", "")
    bridge_port = bridge_cfg.get("port", 0)
    log.info(
        "data_side=%s: HDF5 -> %s; bridge %s:%s (run tb-record-real on this machine).",
        side,
        dataset_dir,
        bridge_host,
        bridge_port,
    )
    if side == DATA_SIDE_SLAVE and not _is_loopback_host(str(bridge_host)):
        log.warning(
            "data_side=slave but bridge host is %s (not loopback). "
            "Images will cross the network; prefer 127.0.0.1 on the vehicle PC.",
            bridge_host,
        )
    if side == DATA_SIDE_HOST and _is_loopback_host(str(bridge_host)):
        log.warning(
            "data_side=host with bridge host 127.0.0.1: HDF5 and gateway are on the same "
            "machine (OK for dev). For split deploy set EXCAVATOR_BRIDGE_HOST to the vehicle IP."
        )


def validate_data_side_for_bridge_tcp(
    side: DataSide | None,
    *,
    backend_mode: str,
    state_reader_mode: str,
    bridge_host: str,
) -> None:
    if side is None:
        return
    uses_tcp = backend_mode == "bridge_tcp" or state_reader_mode == "bridge_tcp"
    if not uses_tcp:
        return
    if side == DATA_SIDE_SLAVE and not _is_loopback_host(bridge_host):
        log.warning(
            "data_side=slave with remote bridge %s: run tb-record-real on the vehicle and "
            "use --bridge-host 127.0.0.1.",
            bridge_host,
        )


def _is_loopback_host(host: str) -> bool:
    value = host.strip().lower()
    if value in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        addr = sockaddr[0]
        if addr == "::1" or addr.startswith("127."):
            return True
    return False
