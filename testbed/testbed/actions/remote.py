"""Remote teleop action transport for host-joystick / slave-recording runs."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from testbed.actions.base import ActionInfo, ActionSource
from testbed.backends.real.contracts import (
    REAL_ACTION_DIM,
    STATUS_TOGGLE_BIT_COUNT,
    as_real_action,
)

log = logging.getLogger(__name__)

REMOTE_ACTION_PROTOCOL_VERSION = 1
REMOTE_ACTION_UPDATE = "remote_action.update"
REMOTE_ACTION_STATUS = "remote_action.status"
DEFAULT_REMOTE_ACTION_PORT = 8770
DEFAULT_REMOTE_ACTION_TIMEOUT_MS = 200.0
REMOTE_ACTION_EVENT_QUEUE_LIMIT = 64


class RemoteActionProtocolError(RuntimeError):
    """Raised when a remote action message is malformed."""


@dataclass(frozen=True)
class RemoteActionPacket:
    seq: int
    action: np.ndarray
    host_sample_time_ns: int
    source_id: str
    toggle_mask: int = 0
    reset_requested: bool = False
    discard_requested: bool = False
    quit_requested: bool = False
    record_start_requested: bool = False
    policy_start_requested: bool = False
    go_home_requested: bool = False


@dataclass(frozen=True)
class RemoteReceiverStatus:
    payload: dict[str, Any]
    receive_time_ns: int


@dataclass
class _LatestRemoteAction:
    packet: RemoteActionPacket
    receive_time_ns: int


@dataclass(frozen=True)
class _PendingRemoteEvent:
    toggle_mask: int
    reset_requested: bool
    discard_requested: bool
    quit_requested: bool
    record_start_requested: bool
    policy_start_requested: bool
    go_home_requested: bool
    receive_time_ns: int


def encode_remote_action_update(
    *,
    seq: int,
    action: Any,
    host_sample_time_ns: int,
    source_id: str,
    toggle_mask: int = 0,
    reset_requested: bool = False,
    discard_requested: bool = False,
    quit_requested: bool = False,
    record_start_requested: bool = False,
    policy_start_requested: bool = False,
    go_home_requested: bool = False,
) -> bytes:
    action4 = as_real_action(action, clip=True)
    payload = {
        "seq": int(seq),
        "action": action4.astype(np.float32).tolist(),
        "host_sample_time_ns": int(host_sample_time_ns),
        "source_id": str(source_id),
        "toggle_mask": _normalize_toggle_mask(toggle_mask),
        "reset_requested": bool(reset_requested),
        "discard_requested": bool(discard_requested),
        "quit_requested": bool(quit_requested),
        "record_start_requested": bool(record_start_requested),
        "policy_start_requested": bool(policy_start_requested),
        "go_home_requested": bool(go_home_requested),
    }
    message = {
        "version": REMOTE_ACTION_PROTOCOL_VERSION,
        "type": REMOTE_ACTION_UPDATE,
        "payload": payload,
    }
    return json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


def encode_remote_receiver_status(payload: Mapping[str, Any]) -> bytes:
    message = {
        "version": REMOTE_ACTION_PROTOCOL_VERSION,
        "type": REMOTE_ACTION_STATUS,
        "payload": dict(payload),
    }
    return json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


def decode_remote_receiver_status(frame: bytes | str) -> dict[str, Any]:
    text = frame.decode("utf-8") if isinstance(frame, bytes) else str(frame)
    try:
        message = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise RemoteActionProtocolError(f"invalid remote status JSON: {exc}") from exc
    if not isinstance(message, Mapping):
        raise RemoteActionProtocolError("remote status frame must be a JSON object")
    version = int(message.get("version", -1))
    if version != REMOTE_ACTION_PROTOCOL_VERSION:
        raise RemoteActionProtocolError(
            f"unsupported remote action protocol version {version}; "
            f"expected {REMOTE_ACTION_PROTOCOL_VERSION}"
        )
    if message.get("type") != REMOTE_ACTION_STATUS:
        raise RemoteActionProtocolError(
            f"unexpected remote status message type {message.get('type')!r}"
        )
    payload = message.get("payload", {})
    if not isinstance(payload, Mapping):
        raise RemoteActionProtocolError("remote status payload must be a JSON object")
    return dict(payload)


def decode_remote_action_update(frame: bytes | str) -> RemoteActionPacket:
    text = frame.decode("utf-8") if isinstance(frame, bytes) else str(frame)
    try:
        message = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise RemoteActionProtocolError(f"invalid remote action JSON: {exc}") from exc
    if not isinstance(message, Mapping):
        raise RemoteActionProtocolError("remote action frame must be a JSON object")
    version = int(message.get("version", -1))
    if version != REMOTE_ACTION_PROTOCOL_VERSION:
        raise RemoteActionProtocolError(
            f"unsupported remote action protocol version {version}; "
            f"expected {REMOTE_ACTION_PROTOCOL_VERSION}"
        )
    if message.get("type") != REMOTE_ACTION_UPDATE:
        raise RemoteActionProtocolError(
            f"unexpected remote action message type {message.get('type')!r}"
        )
    payload = message.get("payload", {})
    if not isinstance(payload, Mapping):
        raise RemoteActionProtocolError("remote action payload must be a JSON object")

    try:
        seq = int(payload.get("seq", -1))
        action = as_real_action(payload.get("action", []), clip=True)
        host_sample_time_ns = int(payload.get("host_sample_time_ns", 0))
    except (TypeError, ValueError) as exc:
        raise RemoteActionProtocolError(str(exc)) from exc
    if seq < 0:
        raise RemoteActionProtocolError("remote action seq must be non-negative")
    if host_sample_time_ns < 0:
        raise RemoteActionProtocolError(
            "remote action host_sample_time_ns must be non-negative"
        )

    return RemoteActionPacket(
        seq=seq,
        action=action.astype(np.float32, copy=True),
        host_sample_time_ns=host_sample_time_ns,
        source_id=str(payload.get("source_id", "remote")),
        toggle_mask=_normalize_toggle_mask(payload.get("toggle_mask", 0)),
        reset_requested=bool(payload.get("reset_requested", False)),
        discard_requested=bool(payload.get("discard_requested", False)),
        quit_requested=bool(payload.get("quit_requested", False)),
        record_start_requested=bool(payload.get("record_start_requested", False)),
        policy_start_requested=bool(payload.get("policy_start_requested", False)),
        go_home_requested=bool(payload.get("go_home_requested", False)),
    )


class RemoteActionClient:
    """TCP writer used by the host-side teleop CLI."""

    def __init__(
        self,
        *,
        host: str,
        port: int = DEFAULT_REMOTE_ACTION_PORT,
        timeout_s: float = 1.0,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None
        self._lock = threading.RLock()
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._latest_status: RemoteReceiverStatus | None = None

    def connect(self) -> None:
        with self._lock:
            if self._sock is not None:
                return
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            sock.settimeout(self.timeout_s)
            self._sock = sock
            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._read_status_loop,
                args=(sock,),
                name="remote-action-status-reader",
                daemon=True,
            )
            self._reader_thread.start()

    def send_update(
        self,
        *,
        seq: int,
        action: Any,
        host_sample_time_ns: int,
        source_id: str,
        toggle_mask: int = 0,
        reset_requested: bool = False,
        discard_requested: bool = False,
        quit_requested: bool = False,
        record_start_requested: bool = False,
        policy_start_requested: bool = False,
        go_home_requested: bool = False,
    ) -> None:
        frame = encode_remote_action_update(
            seq=seq,
            action=action,
            host_sample_time_ns=host_sample_time_ns,
            source_id=source_id,
            toggle_mask=toggle_mask,
            reset_requested=reset_requested,
            discard_requested=discard_requested,
            quit_requested=quit_requested,
            record_start_requested=record_start_requested,
            policy_start_requested=policy_start_requested,
            go_home_requested=go_home_requested,
        )
        with self._lock:
            self.connect()
            assert self._sock is not None
            self._sock.sendall(frame)

    def close(self) -> None:
        with self._lock:
            self._reader_stop.set()
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._sock.close()
                self._sock = None
            reader_thread = self._reader_thread
            self._reader_thread = None
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)

    def latest_status(self) -> RemoteReceiverStatus | None:
        with self._lock:
            return self._latest_status

    def _read_status_loop(self, sock: socket.socket) -> None:
        buffer = b""
        while not self._reader_stop.is_set():
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    payload = decode_remote_receiver_status(line)
                except RemoteActionProtocolError as exc:
                    log.debug("Ignoring non-status remote frame on client: %s", exc)
                    continue
                with self._lock:
                    self._latest_status = RemoteReceiverStatus(
                        payload=payload,
                        receive_time_ns=time.time_ns(),
                    )

    def __enter__(self) -> "RemoteActionClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RemoteActionSource(ActionSource):
    """TCP action source used by the slave-side recorder."""

    def __init__(
        self,
        *,
        bind_host: str = "0.0.0.0",
        port: int = DEFAULT_REMOTE_ACTION_PORT,
        timeout_ms: float = DEFAULT_REMOTE_ACTION_TIMEOUT_MS,
        source_id: str = "remote_teleop",
    ) -> None:
        self.bind_host = str(bind_host)
        self.requested_port = int(port)
        self.timeout_ms = float(timeout_ms)
        self.source_id = str(source_id)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._latest: _LatestRemoteAction | None = None
        self._drop_count = 0
        self._connected = False
        self._pending_events: deque[_PendingRemoteEvent] = deque()
        self._client_sock: socket.socket | None = None
        self._send_lock = threading.RLock()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind_host, self.requested_port))
        server.listen(1)
        server.settimeout(0.2)
        self._server_sock = server
        self.port = int(server.getsockname()[1])
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="remote-action-source",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "RemoteActionSource listening on %s:%d timeout_ms=%.1f",
            self.bind_host,
            self.port,
            self.timeout_ms,
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "RemoteActionSource":
        config = dict(cfg or {})
        return cls(
            bind_host=str(config.get("bind_host", "0.0.0.0")),
            port=int(config.get("port", DEFAULT_REMOTE_ACTION_PORT)),
            timeout_ms=float(
                config.get("timeout_ms", DEFAULT_REMOTE_ACTION_TIMEOUT_MS)
            ),
            source_id=str(config.get("source_id", "remote_teleop")),
        )

    def reset(self) -> None:
        with self._lock:
            self._latest = None
            self._drop_count = 0
            self._clear_pending_events_locked()

    def next_action(self, obs: dict) -> tuple[np.ndarray, ActionInfo]:
        now_ns = time.time_ns()
        with self._lock:
            latest = self._latest
            drop_count = self._drop_count
            connected = self._connected

        if latest is None:
            extras = self._diagnostic_extras(
                seq=-1,
                host_sample_time_ns=0,
                receive_time_ns=0,
                age_ms=0.0,
                stale=True,
                drop_count=drop_count,
                connected=connected,
            )
            return np.zeros(REAL_ACTION_DIM, dtype=np.float32), ActionInfo(
                source_type="teleop",
                source_id=f"remote:{self.source_id}:missing",
                latency_ms=0.0,
                extras=extras,
            )

        packet = latest.packet
        age_ms = max(0.0, (now_ns - latest.receive_time_ns) / 1_000_000.0)
        stale = age_ms > self.timeout_ms
        pending_event = self._pop_pending_event(now_ns=now_ns, stale=stale)
        action = (
            np.zeros(REAL_ACTION_DIM, dtype=np.float32)
            if stale
            else packet.action.astype(np.float32, copy=True)
        )
        extras = self._diagnostic_extras(
            seq=packet.seq,
            host_sample_time_ns=packet.host_sample_time_ns,
            receive_time_ns=latest.receive_time_ns,
            age_ms=age_ms,
            stale=stale,
            drop_count=drop_count,
            connected=connected,
        )
        extras.update(
            {
                "action_timestamp_ns": int(latest.receive_time_ns),
                "toggle_mask": int(pending_event.toggle_mask) if pending_event else 0,
                "reset_requested": (
                    bool(pending_event.reset_requested) if pending_event else False
                ),
                "discard_requested": (
                    bool(pending_event.discard_requested) if pending_event else False
                ),
                "quit_requested": (
                    bool(pending_event.quit_requested) if pending_event else False
                ),
                "record_start_requested": (
                    bool(pending_event.record_start_requested) if pending_event else False
                ),
                "policy_start_requested": (
                    bool(pending_event.policy_start_requested) if pending_event else False
                ),
                "go_home_requested": (
                    bool(pending_event.go_home_requested) if pending_event else False
                ),
            }
        )
        return action, ActionInfo(
            source_type="teleop",
            source_id=f"remote:{packet.source_id}",
            latency_ms=age_ms,
            extras=extras,
        )

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._server_sock.close()
        except OSError:
            pass
        with self._lock:
            client = self._client_sock
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        self._thread.join(timeout=1.0)

    def publish_status(self, payload: Mapping[str, Any]) -> None:
        frame = encode_remote_receiver_status(payload)
        with self._lock:
            client = self._client_sock
        if client is None:
            return
        with self._send_lock:
            try:
                client.sendall(frame)
            except OSError:
                return

    @staticmethod
    def _diagnostic_extras(
        *,
        seq: int,
        host_sample_time_ns: int,
        receive_time_ns: int,
        age_ms: float,
        stale: bool,
        drop_count: int,
        connected: bool,
    ) -> dict[str, Any]:
        return {
            "remote_action_seq": int(seq),
            "remote_action_host_sample_ns": int(host_sample_time_ns),
            "remote_action_receive_ns": int(receive_time_ns),
            "remote_action_age_ms": float(age_ms),
            "remote_action_stale": int(bool(stale)),
            "remote_action_drop_count": int(drop_count),
            "remote_action_connected": int(bool(connected)),
        }

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                client, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            log.info("Remote action client connected from %s", addr)
            client.settimeout(0.2)
            with self._lock:
                self._client_sock = client
                self._connected = True
                self._latest = None
                self._clear_pending_events_locked()
            try:
                self._serve_client(client)
            finally:
                with self._lock:
                    if self._client_sock is client:
                        self._client_sock = None
                    self._connected = False
                try:
                    client.close()
                except OSError:
                    pass
                log.info("Remote action client disconnected")

    def _serve_client(self, client: socket.socket) -> None:
        buffer = b""
        last_seq = -1
        while not self._stop_event.is_set():
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    packet = decode_remote_action_update(line)
                except RemoteActionProtocolError as exc:
                    log.warning("Ignoring malformed remote action: %s", exc)
                    continue
                last_seq = self._store_packet(packet, last_seq=last_seq)

    def _store_packet(self, packet: RemoteActionPacket, *, last_seq: int) -> int:
        with self._lock:
            if packet.seq <= last_seq:
                self._drop_count += 1
                return last_seq
            if last_seq >= 0 and packet.seq > last_seq + 1:
                self._drop_count += int(packet.seq - last_seq - 1)
            receive_time_ns = time.time_ns()
            self._latest = _LatestRemoteAction(
                packet=packet,
                receive_time_ns=receive_time_ns,
            )
            if (
                packet.toggle_mask
                or packet.reset_requested
                or packet.discard_requested
                or packet.quit_requested
                or packet.record_start_requested
                or packet.policy_start_requested
                or packet.go_home_requested
            ):
                if len(self._pending_events) >= REMOTE_ACTION_EVENT_QUEUE_LIMIT:
                    self._pending_events.popleft()
                    self._drop_count += 1
                self._pending_events.append(
                    _PendingRemoteEvent(
                        toggle_mask=int(packet.toggle_mask),
                        reset_requested=bool(packet.reset_requested),
                        discard_requested=bool(packet.discard_requested),
                        quit_requested=bool(packet.quit_requested),
                        record_start_requested=bool(packet.record_start_requested),
                        policy_start_requested=bool(packet.policy_start_requested),
                        go_home_requested=bool(packet.go_home_requested),
                        receive_time_ns=receive_time_ns,
                    )
                )
            return int(packet.seq)

    def _pop_pending_event(
        self,
        *,
        now_ns: int,
        stale: bool,
    ) -> _PendingRemoteEvent | None:
        with self._lock:
            while self._pending_events:
                event = self._pending_events[0]
                event_age_ms = max(
                    0.0, (int(now_ns) - int(event.receive_time_ns)) / 1_000_000.0
                )
                if event_age_ms > self.timeout_ms:
                    self._pending_events.popleft()
                    continue
                if stale:
                    return None
                return self._pending_events.popleft()
        return None

    def _clear_pending_events_locked(self) -> None:
        self._pending_events.clear()


def _normalize_toggle_mask(value: Any) -> int:
    mask = int(value or 0)
    max_mask = (1 << STATUS_TOGGLE_BIT_COUNT) - 1
    return int(mask & max_mask)
