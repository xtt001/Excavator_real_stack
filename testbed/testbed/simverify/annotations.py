"""Observable-only cycle and sector annotation primitives for SimVerify M0.

The functions in this module deliberately accept only image features, qpos,
qvel, action, and ``step_id``.  They do not know about ``env_state`` or the
PACT ``v2`` label tree.  Privileged comparisons belong in a physically
separate post-hoc oracle audit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ANNOTATION_SCHEMA = "observable_cycle_annotation_v2"
ANNOTATION_THRESHOLDS_SCHEMA = "observable_annotation_thresholds_v2"
SECTORS = ("left", "center", "right")
SECTOR_TO_INDEX = {name: index for index, name in enumerate(SECTORS)}


@dataclass(frozen=True)
class EpisodeSignals:
    """The complete set of numeric signals allowed to drive annotation."""

    episode_id: int
    step_id: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    dt: float

    def validate(self) -> None:
        count = int(self.step_id.shape[0])
        if count <= 0:
            raise ValueError(f"episode_{self.episode_id}: empty episode")
        if self.step_id.shape != (count,):
            raise ValueError(f"episode_{self.episode_id}: invalid step_id shape")
        for name, value in (
            ("qpos", self.qpos),
            ("qvel", self.qvel),
            ("action", self.action),
        ):
            if value.shape != (count, 4):
                raise ValueError(
                    f"episode_{self.episode_id}: {name} shape {value.shape}, "
                    f"expected {(count, 4)}"
                )
            if not np.isfinite(value).all():
                raise ValueError(
                    f"episode_{self.episode_id}: {name} contains non-finite values"
                )
        if not np.array_equal(
            self.step_id,
            np.arange(int(self.step_id[0]), int(self.step_id[0]) + count),
        ):
            raise ValueError(
                f"episode_{self.episode_id}: step_id must be strictly contiguous"
            )
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"episode_{self.episode_id}: invalid dt={self.dt!r}")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs where a one-dimensional boolean mask is true."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    edges = np.flatnonzero(
        np.concatenate(
            (
                np.asarray([True]),
                values[1:] != values[:-1],
                np.asarray([True]),
            )
        )
    )
    return [
        (int(start), int(end))
        for start, end in zip(edges[:-1], edges[1:])
        if bool(values[start])
    ]


def fit_1d_kmeans(
    values: np.ndarray,
    *,
    clusters: int,
    max_iterations: int = 200,
) -> np.ndarray:
    """Fit deterministic one-dimensional k-means without a sklearn dependency."""

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < clusters:
        raise ValueError(
            f"need at least {clusters} finite samples, got {samples.size}"
        )
    centers = np.quantile(
        samples,
        np.linspace(0.1, 0.9, int(clusters), dtype=np.float64),
    )
    for _ in range(max_iterations):
        labels = np.argmin(
            np.abs(samples[:, None] - centers[None, :]),
            axis=1,
        )
        updated = np.asarray(
            [
                np.mean(samples[labels == index])
                if np.any(labels == index)
                else centers[index]
                for index in range(clusters)
            ],
            dtype=np.float64,
        )
        updated.sort()
        if np.allclose(updated, centers, rtol=0.0, atol=1e-12):
            centers = updated
            break
        centers = updated
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError(f"1D clusters are not identifiable: {centers.tolist()}")
    return centers


def _release_runs(
    episode: EpisodeSignals,
    *,
    dump_swing_threshold: float,
    action_deadzone: float,
    minimum_steps: int,
) -> list[tuple[int, int]]:
    active = (
        np.asarray(episode.action[:, 3], dtype=np.float64) < -action_deadzone
    ) & (
        np.asarray(episode.qpos[:, 0], dtype=np.float64)
        > dump_swing_threshold
    )
    return [
        (start, end)
        for start, end in _runs(active)
        if end - start >= int(minimum_steps)
    ]


def _merge_release_runs(
    runs: Sequence[tuple[int, int]],
    *,
    swing_qpos: np.ndarray,
    dump_swing_threshold: float,
) -> list[tuple[int, int]]:
    """Merge release pulses until the observable swing exits the dump cluster.

    There is deliberately no learned gap-duration threshold. Episode bootstrap
    does not identify a stable short/long gap boundary, while a swing-cluster
    exit already separates two dumps using an observable, data-fitted state.
    """

    merged: list[list[int]] = []
    for start, end in runs:
        if not merged:
            merged.append([int(start), int(end)])
            continue
        previous_end = merged[-1][1]
        remains_at_dump = bool(
            int(start) >= int(previous_end)
            and np.min(swing_qpos[previous_end : int(start) + 1])
            > dump_swing_threshold
        )
        if remains_at_dump:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    return [(start, end) for start, end in merged]


def fit_numeric_annotation_thresholds(
    episodes: Sequence[EpisodeSignals],
    *,
    action_deadzone: float,
) -> dict[str, Any]:
    """Fit dump/release thresholds using train episodes only.

    ``minimum_release_steps`` has a structural floor of three source rows so a
    one-tick command spike can never become an event.  All numerical separation
    thresholds are then generated from the supplied episode distribution.
    """

    if not episodes:
        raise ValueError("at least one train episode is required")
    for episode in episodes:
        episode.validate()
    dts = np.asarray([episode.dt for episode in episodes], dtype=np.float64)
    if not np.allclose(dts, dts[0], rtol=0.0, atol=1e-8):
        raise ValueError(f"source dt is inconsistent: {dts.tolist()}")

    release_swing: list[np.ndarray] = []
    for episode in episodes:
        mask = episode.action[:, 3] < -float(action_deadzone)
        release_swing.append(
            np.asarray(episode.qpos[mask, 0], dtype=np.float64)
        )
    release_values = np.concatenate(release_swing)
    dump_centers = fit_1d_kmeans(release_values, clusters=2)
    dump_threshold = float(np.mean(dump_centers))
    return {
        "schema": ANNOTATION_THRESHOLDS_SCHEMA,
        "fit_scope": "train_only_numeric_observations",
        "observable_inputs": ["qpos", "qvel", "action", "step_id"],
        "privilege_used": False,
        "source_dt_s": float(dts[0]),
        "action_deadzone": float(action_deadzone),
        "dump_release": {
            "swing_cluster_centers": dump_centers.tolist(),
            "swing_threshold": dump_threshold,
            "minimum_release_steps": 3,
            "merge_rule": "merge_until_swing_exits_dump_cluster",
            "gap_duration_threshold": None,
        },
        "ready": {
            "activity": "abs_swing_qvel_plus_abs_swing_action",
            "minimum_envelope_steps": 1,
            "search_end": "first_sustained_positive_bucket_or_low_swing_run_end",
            "local_basin_rule": (
                "contiguous_run_containing_local_activity_minimum_below_"
                "midpoint_of_local_minimum_and_local_median"
            ),
            "allows_nonzero_qvel": True,
        },
    }


def bootstrap_numeric_thresholds(
    episodes: Sequence[EpisodeSignals],
    *,
    action_deadzone: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Episode-level bootstrap stability for the numeric candidate rules."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(int(seed))
    dump_thresholds: list[float] = []
    dump_centers: list[list[float]] = []
    failures = 0
    for _ in range(int(samples)):
        draw = [
            episodes[index]
            for index in rng.integers(0, len(episodes), size=len(episodes))
        ]
        try:
            fitted = fit_numeric_annotation_thresholds(
                draw,
                action_deadzone=action_deadzone,
            )
        except ValueError:
            failures += 1
            continue
        release = fitted["dump_release"]
        dump_thresholds.append(float(release["swing_threshold"]))
        dump_centers.append(
            list(map(float, release["swing_cluster_centers"]))
        )
    if not dump_thresholds:
        raise ValueError("all numeric-threshold bootstrap samples failed")

    def summary(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(array)),
            "p02_5": float(np.quantile(array, 0.025)),
            "p97_5": float(np.quantile(array, 0.975)),
            "std": float(np.std(array)),
        }

    center_array = np.asarray(dump_centers, dtype=np.float64)
    return {
        "unit": "source_episode",
        "seed": int(seed),
        "requested_samples": int(samples),
        "successful_samples": len(dump_thresholds),
        "failed_samples": int(failures),
        "dump_swing_cluster_centers": {
            "median": np.median(center_array, axis=0).tolist(),
            "p02_5": np.quantile(center_array, 0.025, axis=0).tolist(),
            "p97_5": np.quantile(center_array, 0.975, axis=0).tolist(),
            "std": np.std(center_array, axis=0).tolist(),
        },
        "dump_swing_threshold": summary(dump_thresholds),
        "release_pulse_merge_rule": (
            "structural_observable_swing_cluster_exit_no_gap_threshold"
        ),
    }


def detect_dump_releases(
    episode: EpisodeSignals,
    thresholds: Mapping[str, Any],
) -> list[tuple[int, int]]:
    release = thresholds["dump_release"]
    runs = _release_runs(
        episode,
        dump_swing_threshold=float(release["swing_threshold"]),
        action_deadzone=float(thresholds["action_deadzone"]),
        minimum_steps=int(release["minimum_release_steps"]),
    )
    return _merge_release_runs(
        runs,
        swing_qpos=np.asarray(episode.qpos[:, 0], dtype=np.float64),
        dump_swing_threshold=float(release["swing_threshold"]),
    )


def _longest_low_swing_run(
    swing_qpos: np.ndarray,
    *,
    start: int,
    end: int,
    dump_swing_threshold: float,
    minimum_steps: int,
) -> tuple[int, int] | None:
    if end <= start:
        return None
    runs = [
        (start + lo, start + hi)
        for lo, hi in _runs(
            np.asarray(swing_qpos[start:end]) < dump_swing_threshold
        )
        if hi - lo >= int(minimum_steps)
    ]
    if not runs:
        return None
    return max(runs, key=lambda item: (item[1] - item[0], -item[0]))


def _readiness_score(episode: EpisodeSignals) -> np.ndarray:
    return np.abs(episode.qvel[:, 0]) + np.abs(
        episode.action[:, 0],
    )


def _ready_envelope(
    episode: EpisodeSignals,
    *,
    start: int,
    end: int,
    dump_swing_threshold: float,
    action_deadzone: float,
    minimum_steps: int,
) -> dict[str, Any] | None:
    low_run = _longest_low_swing_run(
        episode.qpos[:, 0],
        start=int(start),
        end=int(end),
        dump_swing_threshold=float(dump_swing_threshold),
        minimum_steps=int(minimum_steps),
    )
    if low_run is None:
        return None
    low_start, low_end = low_run
    dig_entry = _first_sustained_positive_bucket(
        episode,
        start=low_start,
        end=low_end,
        action_deadzone=action_deadzone,
        minimum_steps=minimum_steps,
    )
    search_end = low_end if dig_entry is None else int(dig_entry[0])
    if search_end <= low_start:
        return None
    score = _readiness_score(episode)
    local_score = score[low_start:search_end]
    representative = int(low_start + np.argmin(local_score))
    local_minimum = float(np.min(local_score))
    local_median = float(np.median(local_score))
    activity_threshold = float((local_minimum + local_median) / 2.0)
    candidate_runs = [
        (low_start + begin, low_start + finish)
        for begin, finish in _runs(
            local_score <= float(activity_threshold)
        )
    ]
    containing = [
        interval
        for interval in candidate_runs
        if interval[0] <= representative < interval[1]
    ]
    interval_start, interval_end = (
        containing[0]
        if containing
        else (representative, representative + 1)
    )
    return {
        "interval": [int(interval_start), int(interval_end)],
        "representative_step": int(representative),
        "low_swing_run": [int(low_start), int(low_end)],
        "readiness_score": float(score[representative]),
        "activity_threshold": float(activity_threshold),
        "activity_basin_contrast": float(local_median - local_minimum),
        "allows_nonzero_qvel": True,
    }


def _first_sustained_positive_bucket(
    episode: EpisodeSignals,
    *,
    start: int,
    end: int,
    action_deadzone: float,
    minimum_steps: int,
) -> tuple[int, int] | None:
    """Return the first full sustained positive-bucket run, if observable."""

    mask = episode.action[start:end, 3] > action_deadzone
    runs = [
        (start + lo, start + hi)
        for lo, hi in _runs(mask)
        if hi - lo >= minimum_steps
    ]
    return runs[0] if runs else None


def _carry_transition(
    episode: EpisodeSignals,
    *,
    start: int,
    dump_start: int,
    dump_swing_threshold: float,
) -> int:
    if dump_start <= start:
        return int(start)
    swing = np.asarray(episode.qpos[start:dump_start, 0], dtype=np.float64)
    crossings = np.flatnonzero(swing >= dump_swing_threshold)
    if crossings.size:
        return int(start + crossings[0])
    # Fall back to the strongest positive swing speed, still observable-only.
    return int(start + np.argmax(episode.qvel[start:dump_start, 0]))


def annotate_numeric_cycles(
    episode: EpisodeSignals,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create observable cycle candidates before visual-sector fusion."""

    episode.validate()
    releases = detect_dump_releases(episode, thresholds)
    if not releases:
        return []
    release = thresholds["dump_release"]
    dump_threshold = float(release["swing_threshold"])
    minimum_steps = int(release["minimum_release_steps"])
    action_deadzone = float(thresholds["action_deadzone"])
    ready_minimum_steps = int(thresholds["ready"]["minimum_envelope_steps"])
    count = int(episode.step_id.size)

    gap_ready: list[dict[str, Any] | None] = []
    boundaries = [0] + [end for _, end in releases]
    following = [start for start, _ in releases] + [count]
    for start, end in zip(boundaries, following):
        gap_ready.append(
            _ready_envelope(
                episode,
                start=int(start),
                end=int(end),
                dump_swing_threshold=dump_threshold,
                action_deadzone=action_deadzone,
                minimum_steps=ready_minimum_steps,
            )
        )

    records: list[dict[str, Any]] = []
    for index, (dump_start, dump_end) in enumerate(releases):
        ready_start = gap_ready[index]
        ready_end = gap_ready[index + 1]
        reasons: list[str] = []
        if ready_start is None:
            reasons.append("ready_start_not_identifiable")
        if ready_end is None:
            reasons.append("ready_end_not_identifiable")
        cycle_start = (
            int(dump_start)
            if ready_start is None
            else int(ready_start["representative_step"])
        )
        cycle_end = (
            int(dump_end)
            if ready_end is None
            else int(ready_end["representative_step"])
        )
        dig_entry_interval: tuple[int, int] | None = None
        carry: int | None = None
        if ready_start is not None:
            representative = int(ready_start["representative_step"])
            dig_entry_interval = _first_sustained_positive_bucket(
                episode,
                start=representative,
                end=dump_start,
                action_deadzone=action_deadzone,
                minimum_steps=minimum_steps,
            )
            if dig_entry_interval is not None:
                carry = _carry_transition(
                    episode,
                    start=int(dig_entry_interval[0]),
                    dump_start=dump_start,
                    dump_swing_threshold=dump_threshold,
                )
        if dig_entry_interval is None:
            reasons.append("dig_entry_proxy_not_identifiable")

        event_order = [cycle_start]
        if dig_entry_interval is not None:
            event_order.append(int(dig_entry_interval[0]))
        if carry is not None:
            event_order.append(int(carry))
        event_order.extend((int(dump_start), int(dump_end - 1)))
        if ready_end is not None:
            event_order.append(cycle_end)
        if any(
            event_order[position] > event_order[position + 1]
            for position in range(len(event_order) - 1)
        ):
            reasons.append("observable_event_order_invalid")

        dig_entry_proxy = (
            None
            if dig_entry_interval is None
            else {
                "interval": [
                    int(dig_entry_interval[0]),
                    int(dig_entry_interval[1]),
                ],
                "representative_step": int(dig_entry_interval[0]),
            }
        )
        carry_transition_proxy = (
            None
            if carry is None
            else {
                "interval": [
                    max(0, int(carry - minimum_steps)),
                    min(count, int(carry + minimum_steps + 1)),
                ],
                "representative_step": int(carry),
            }
        )
        current_sector_valid = dig_entry_proxy is not None

        records.append(
            {
                "schema": ANNOTATION_SCHEMA,
                "episode_id": int(episode.episode_id),
                "cycle_id": int(index),
                "command": {
                    "current_sector": "unknown_not_recorded",
                    "next_ready_sector": "unknown_not_recorded",
                },
                "condition_source": "hindsight_outcome",
                "source_steps": [int(cycle_start), int(cycle_end)],
                "observable_events": {
                    "ready_start": ready_start,
                    "dig_entry_proxy": dig_entry_proxy,
                    "carry_transition_proxy": carry_transition_proxy,
                    "dump_start_proxy": {
                        "interval": [int(dump_start), int(dump_start + minimum_steps)],
                        "representative_step": int(dump_start),
                    },
                    "dump_end_proxy": {
                        "interval": [int(dump_end - minimum_steps), int(dump_end)],
                        "representative_step": int(dump_end - 1),
                    },
                    "ready_end": ready_end,
                },
                "numeric_sector_evidence": {
                    "current_swing_qpos": None,
                    "next_swing_qpos": None,
                },
                "sector_observations": {"current": None, "next": None},
                "sector_validity": {
                    "current": {
                        "valid": bool(current_sector_valid),
                        "source_cycle_id": int(index),
                        "reason_codes": (
                            []
                            if current_sector_valid
                            else ["dig_entry_proxy_not_identifiable"]
                        ),
                    },
                    "next": None,
                },
                "policy_condition": {
                    "current_sector": None,
                    "next_ready_sector": None,
                    "vector": None,
                },
                "outcome": {
                    "actual_current_sector": None,
                    "actual_next_ready_sector": None,
                },
                "quality": {
                    "status": "numeric_candidate" if not reasons else "ambiguous",
                    "confidence": None,
                    "review_required": bool(reasons),
                    "reason_codes": reasons,
                },
                "verification": {
                    "privilege_used_for_annotation": False,
                    "visual_confirmation_complete": False,
                },
            }
        )

    for record in records:
        current_event = record["observable_events"]["dig_entry_proxy"]
        if current_event is None:
            continue
        current_observation = _sector_observation(
            current_event,
            episode.qpos,
        )
        record["sector_observations"]["current"] = current_observation
        record["numeric_sector_evidence"]["current_swing_qpos"] = float(
            current_observation["swing_qpos_at_representative"]
        )

    for index, record in enumerate(records):
        next_record = records[index + 1] if index + 1 < len(records) else None
        adjacent = (
            next_record is not None
            and int(next_record["cycle_id"]) == int(record["cycle_id"]) + 1
        )
        if adjacent:
            next_validity = next_record["sector_validity"]["current"]
            record["sector_validity"]["next"] = {
                "valid": bool(next_validity["valid"]),
                "source_cycle_id": int(next_validity["source_cycle_id"]),
                "reason_codes": list(next_validity["reason_codes"]),
            }
            if bool(next_validity["valid"]):
                next_observation = dict(
                    next_record["sector_observations"]["current"]
                )
                record["sector_observations"]["next"] = next_observation
                record["numeric_sector_evidence"]["next_swing_qpos"] = float(
                    next_record["numeric_sector_evidence"]["current_swing_qpos"]
                )
        else:
            record["sector_validity"]["next"] = {
                "valid": False,
                "source_cycle_id": None,
                "reason_codes": ["next_cycle_not_available"],
            }
        if not bool(record["sector_validity"]["next"]["valid"]):
            reasons = list(record["quality"]["reason_codes"])
            reasons.append("next_dig_entry_not_observable")
            record["quality"] = {
                "status": "ambiguous",
                "confidence": None,
                "review_required": True,
                "reason_codes": sorted(set(reasons)),
            }
    return records


def _sector_observation(
    event: Mapping[str, Any],
    qpos: np.ndarray,
) -> dict[str, Any]:
    start, end = map(int, event["interval"])
    if end <= start:
        raise ValueError("dig-entry sector interval must be non-empty")
    representative = int(event["representative_step"])
    if not start <= representative < end:
        raise ValueError(
            "dig-entry representative must lie inside its half-open interval"
        )
    return {
        "source": "dig_entry_proxy",
        "interval": [start, end],
        "representative_step": representative,
        "swing_qpos_at_representative": float(qpos[representative, 0]),
    }


def fit_sector_thresholds(
    cycles: Sequence[Mapping[str, Any]],
    *,
    field: str = "current_swing_qpos",
) -> dict[str, Any]:
    values = np.asarray(
        [
            record["numeric_sector_evidence"][field]
            for record in cycles
            if record["numeric_sector_evidence"].get(field) is not None
            and bool(record["sector_validity"]["current"]["valid"])
        ],
        dtype=np.float64,
    )
    centers = fit_1d_kmeans(values, clusters=3)
    boundaries = (centers[:-1] + centers[1:]) / 2.0
    # The frozen PACT physical naming contract uses lower swing -> left,
    # sector_id 0 and higher swing -> right, sector_id 2. This names source
    # clusters only; it is not a sim-to-real numerical equivalence claim.
    return {
        "cluster_centers_low_to_high": centers.tolist(),
        "boundaries_low_to_high": boundaries.tolist(),
        "labels_low_to_high": ["left", "center", "right"],
        "label_orientation": {
            "rule": "lower_swing_left_higher_swing_right",
            "reference_repo": "/home/pingfan/PACT/excavator_testbed",
            "reference_commit": (
                "9bcb29212b59cc3f788ed6c5046677de26c1ee3b"
            ),
            "reference_path": "testbed/planner/corridor_servo.py",
            "real_unit_equivalence_claimed": False,
        },
        "boundary_review_margin": 0.0,
        "boundary_review_margin_source": (
            "pending_source_episode_bootstrap_ci_half_width"
        ),
        "source_representation": "swing_position_norm",
    }


def classify_sector(
    swing_qpos: float,
    thresholds: Mapping[str, Any],
) -> tuple[str | None, float, bool]:
    boundaries = np.asarray(
        thresholds["boundaries_low_to_high"],
        dtype=np.float64,
    )
    centers = np.asarray(
        thresholds["cluster_centers_low_to_high"],
        dtype=np.float64,
    )
    margin = float(thresholds["boundary_review_margin"])
    distance_to_boundary = float(np.min(np.abs(boundaries - float(swing_qpos))))
    boundary = distance_to_boundary <= margin
    index = int(np.searchsorted(boundaries, float(swing_qpos), side="right"))
    label = str(thresholds["labels_low_to_high"][index])
    cluster_gap = float(
        np.min(np.diff(centers))
    )
    confidence = min(1.0, distance_to_boundary / max(cluster_gap, 1e-12))
    return (None if boundary else label, confidence, boundary)


def condition_vector(current_sector: str, next_sector: str) -> list[float]:
    if current_sector not in SECTOR_TO_INDEX:
        raise ValueError(f"invalid current sector: {current_sector!r}")
    if next_sector not in SECTOR_TO_INDEX:
        raise ValueError(f"invalid next sector: {next_sector!r}")
    vector = np.zeros(6, dtype=np.float32)
    vector[SECTOR_TO_INDEX[current_sector]] = 1.0
    vector[3 + SECTOR_TO_INDEX[next_sector]] = 1.0
    return vector.tolist()


def unit_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def fit_visual_sector_centroids(
    feature_rows: Sequence[tuple[int, str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Fit eye-pair centroids from qpos-derived train labels."""

    result: dict[str, np.ndarray] = {}
    for sector in SECTORS:
        rows = [
            np.asarray(feature, dtype=np.float64)
            for _episode_id, label, feature in feature_rows
            if label == sector
        ]
        if not rows:
            raise ValueError(f"no train visual features for sector {sector}")
        result[sector] = unit_normalize(
            np.mean(unit_normalize(np.stack(rows, axis=0)), axis=0)
        )
    return result


def classify_visual_sector(
    feature: np.ndarray,
    centroids: Mapping[str, np.ndarray],
    *,
    minimum_similarity: float,
    minimum_margin: float,
) -> tuple[str | None, float, dict[str, float]]:
    value = unit_normalize(np.asarray(feature, dtype=np.float64).reshape(1, -1))[0]
    scores = {
        sector: float(np.dot(value, unit_normalize(centroids[sector])))
        for sector in SECTORS
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, best = ranked[0]
    margin = float(best - ranked[1][1])
    accepted = best >= float(minimum_similarity) and margin >= float(minimum_margin)
    return (label if accepted else None, float(best), scores)


def fuse_cycle_sectors(
    cycle: dict[str, Any],
    *,
    sector_thresholds: Mapping[str, Any],
    current_eye_feature: np.ndarray | None,
    next_eye_feature: np.ndarray | None,
    visual_centroids: Mapping[str, np.ndarray],
    visual_minimum_similarity: float,
    visual_minimum_margin: float,
) -> dict[str, Any]:
    """Require independent qpos and eye-pair evidence to agree."""

    reasons = list(cycle["quality"]["reason_codes"])
    fused: list[str | None] = []
    visual_evidence: dict[str, Any] = {}
    for role, numeric_field, feature in (
        ("current", "current_swing_qpos", current_eye_feature),
        ("next", "next_swing_qpos", next_eye_feature),
    ):
        numeric_value = cycle["numeric_sector_evidence"].get(numeric_field)
        if numeric_value is None:
            reasons.append(f"{role}_numeric_sector_missing")
            fused.append(None)
            continue
        qpos_label, qpos_confidence, qpos_boundary = classify_sector(
            float(numeric_value),
            sector_thresholds,
        )
        if qpos_boundary:
            reasons.append(f"{role}_qpos_sector_boundary")
        if feature is None:
            reasons.append(f"{role}_eye_feature_missing")
            fused.append(None)
            continue
        visual_label, visual_confidence, scores = classify_visual_sector(
            feature,
            visual_centroids,
            minimum_similarity=visual_minimum_similarity,
            minimum_margin=visual_minimum_margin,
        )
        if visual_label is None:
            reasons.append(f"{role}_visual_sector_ambiguous")
        if (
            qpos_label is not None
            and visual_label is not None
            and qpos_label != visual_label
        ):
            reasons.append(f"{role}_qpos_visual_disagreement")
        label = (
            qpos_label
            if qpos_label is not None and qpos_label == visual_label
            else None
        )
        fused.append(label)
        visual_evidence[role] = {
            "qpos_label": qpos_label,
            "qpos_confidence": qpos_confidence,
            "visual_label": visual_label,
            "visual_confidence": visual_confidence,
            "visual_scores": scores,
            "fused_label": label,
        }

    current, next_sector = fused
    accepted = current is not None and next_sector is not None and not reasons
    cycle["visual_sector_evidence"] = visual_evidence
    cycle["verification"]["visual_confirmation_complete"] = bool(
        current_eye_feature is not None and next_eye_feature is not None
    )
    cycle["outcome"]["actual_current_sector"] = current
    cycle["outcome"]["actual_next_ready_sector"] = next_sector
    cycle["policy_condition"]["current_sector"] = current
    cycle["policy_condition"]["next_ready_sector"] = next_sector
    cycle["policy_condition"]["vector"] = (
        condition_vector(current, next_sector) if accepted else None
    )
    confidences = [
        evidence["qpos_confidence"]
        for evidence in visual_evidence.values()
    ] + [
        evidence["visual_confidence"]
        for evidence in visual_evidence.values()
    ]
    cycle["quality"] = {
        "status": "accepted" if accepted else "ambiguous",
        "confidence": float(min(confidences)) if confidences else 0.0,
        "review_required": not accepted,
        "reason_codes": sorted(set(reasons)),
    }
    return cycle


def map_source_interval_to_target(
    interval: Sequence[int],
    source_indices: np.ndarray,
) -> list[int]:
    """Map a half-open source-row interval to a half-open 20 Hz interval."""

    selected = np.asarray(source_indices, dtype=np.int64)
    start, end = int(interval[0]), int(interval[1])
    target_start = int(np.searchsorted(selected, start, side="left"))
    target_end = int(np.searchsorted(selected, end, side="left"))
    return [target_start, target_end]


def iter_event_representatives(
    cycles: Iterable[Mapping[str, Any]],
) -> Iterable[tuple[int, int, str, int]]:
    for cycle in cycles:
        episode_id = int(cycle["episode_id"])
        cycle_id = int(cycle["cycle_id"])
        for name, event in cycle["observable_events"].items():
            if event is not None:
                yield (
                    episode_id,
                    cycle_id,
                    str(name),
                    int(event["representative_step"]),
                )
