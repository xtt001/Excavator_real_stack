"""Host-side SSH watcher for slave-written real-machine HDF5 episodes."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteEpisode:
    name: str
    size: int
    mtime: float


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="tb-dataset-qc-watch-ssh",
        description="Watch slave-side HDF5 episodes over SSH, copy them to host, and QC locally.",
    )
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-user", default=None)
    parser.add_argument("--remote-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--poll-s", type=float, default=3.0)
    parser.add_argument("--stable-checks", type=int, default=2)
    parser.add_argument("--stable-interval-s", type=float, default=1.0)
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--max-cache-gb", type=float, default=0.0)
    args = parser.parse_args()

    target = _ssh_target(args.ssh_host, args.ssh_user)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir is not None else cache_dir / "qc" / "live"
    processed: set[str] = set()

    log.info(
        "SSH QC watcher starts: %s:%s -> %s",
        target,
        args.remote_dir,
        cache_dir,
    )
    while True:
        try:
            episodes = list_remote_episodes(
                target=target,
                remote_dir=args.remote_dir,
                ssh_options=args.ssh_option,
            )
            for episode in episodes:
                key = f"{episode.name}:{episode.size}:{episode.mtime:.6f}"
                if key in processed:
                    continue
                stable = wait_until_remote_stable(
                    target=target,
                    remote_dir=args.remote_dir,
                    name=episode.name,
                    ssh_options=args.ssh_option,
                    checks=args.stable_checks,
                    interval_s=args.stable_interval_s,
                )
                if stable is None:
                    continue
                local_path = copy_remote_episode(
                    target=target,
                    remote_dir=args.remote_dir,
                    name=episode.name,
                    cache_dir=cache_dir,
                    ssh_options=args.ssh_option,
                )
                from testbed.data.episode_qc import run_episode_qc

                result = run_episode_qc(local_path, output_dir=output_dir)
                status = "OK" if result["ok"] else "ERROR"
                log.info(
                    "QC %s %s errors=%s warnings=%s",
                    status,
                    episode.name,
                    ",".join(result["errors"]) or "-",
                    ",".join(result["warnings"]) or "-",
                )
                processed.add(key)
                if not args.keep_cache:
                    _trim_cache(cache_dir, max_cache_gb=float(args.max_cache_gb))
        except KeyboardInterrupt:
            raise
        except Exception:
            log.exception("SSH QC watcher poll failed")
        if args.once:
            break
        time.sleep(max(0.1, float(args.poll_s)))


def list_remote_episodes(
    *,
    target: str,
    remote_dir: str,
    ssh_options: Iterable[str] = (),
) -> list[RemoteEpisode]:
    command = (
        "find "
        + shlex.quote(remote_dir)
        + " -maxdepth 1 -type f -name 'episode_*.hdf5' "
        + "-printf '%f\\t%s\\t%T@\\n'"
    )
    proc = _run_ssh(target=target, command=command, ssh_options=ssh_options)
    return parse_remote_find_output(proc.stdout)


def wait_until_remote_stable(
    *,
    target: str,
    remote_dir: str,
    name: str,
    ssh_options: Iterable[str] = (),
    checks: int = 2,
    interval_s: float = 1.0,
) -> RemoteEpisode | None:
    checks = max(1, int(checks))
    last: RemoteEpisode | None = None
    stable_count = 0
    while stable_count < checks:
        current = stat_remote_episode(
            target=target,
            remote_dir=remote_dir,
            name=name,
            ssh_options=ssh_options,
        )
        if current is None:
            return None
        if last is not None and current.size == last.size and current.mtime == last.mtime:
            stable_count += 1
        else:
            stable_count = 1
        last = current
        if stable_count < checks:
            time.sleep(max(0.0, float(interval_s)))
    return last


def stat_remote_episode(
    *,
    target: str,
    remote_dir: str,
    name: str,
    ssh_options: Iterable[str] = (),
) -> RemoteEpisode | None:
    remote_path = _remote_path(remote_dir, name)
    command = (
        "test -f "
        + shlex.quote(remote_path)
        + " && stat -c '%n\\t%s\\t%Y' "
        + shlex.quote(remote_path)
    )
    proc = _run_ssh(
        target=target,
        command=command,
        ssh_options=ssh_options,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = proc.stdout.strip().split("\t")
    if len(parts) != 3:
        return None
    return RemoteEpisode(name=Path(parts[0]).name, size=int(parts[1]), mtime=float(parts[2]))


def copy_remote_episode(
    *,
    target: str,
    remote_dir: str,
    name: str,
    cache_dir: Path,
    ssh_options: Iterable[str] = (),
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / name
    tmp_path = cache_dir / f".{name}.tmp.{os.getpid()}"
    if tmp_path.exists():
        tmp_path.unlink()
    remote = f"{target}:{_remote_path(remote_dir, name)}"
    if shutil.which("rsync") is not None:
        cmd = ["rsync", "-a", "--partial"]
        options = list(ssh_options)
        if options:
            cmd.extend(["-e", "ssh " + " ".join(shlex.quote(opt) for opt in options)])
        cmd.extend([remote, str(tmp_path)])
    else:
        cmd = ["scp", *list(ssh_options), remote, str(tmp_path)]
    subprocess.run(cmd, check=True, text=True)
    tmp_path.replace(final_path)
    return final_path


def parse_remote_find_output(text: str) -> list[RemoteEpisode]:
    episodes: list[RemoteEpisode] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name = parts[0]
        if name.startswith(".") or not name.startswith("episode_") or not name.endswith(".hdf5"):
            continue
        try:
            episodes.append(
                RemoteEpisode(name=name, size=int(parts[1]), mtime=float(parts[2]))
            )
        except ValueError:
            continue
    return sorted(episodes, key=lambda item: _episode_sort_key(item.name))


def _run_ssh(
    *,
    target: str,
    command: str,
    ssh_options: Iterable[str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *list(ssh_options), target, command],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _ssh_target(host: str, user: str | None) -> str:
    return str(host) if not user else f"{user}@{host}"


def _remote_path(remote_dir: str, name: str) -> str:
    return str(Path(str(remote_dir).rstrip("/")) / Path(name).name)


def _episode_sort_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    try:
        return int(stem.split("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**31 - 1, name


def _trim_cache(cache_dir: Path, *, max_cache_gb: float) -> None:
    if max_cache_gb <= 0.0:
        return
    files = sorted(
        (p for p in cache_dir.glob("episode_*.hdf5") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    max_bytes = int(max_cache_gb * 1024**3)
    total = sum(p.stat().st_size for p in files)
    while total > max_bytes and files:
        victim = files.pop(0)
        size = victim.stat().st_size
        victim.unlink(missing_ok=True)
        total -= size


if __name__ == "__main__":
    main()
