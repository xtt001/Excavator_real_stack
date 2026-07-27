"""Runtime-safe export and detector for the frozen v11 habit ready boundary."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from testbed.simverify.annotations import classify_sector
from testbed.simverify.artifacts import (
    artifact_identity,
    verify_checksums,
    write_checksums,
    write_json,
)
from testbed.simverify.contracts import git_provenance, sha256_file
from testbed.simverify.features import FrozenResNet18FeatureExtractor
from testbed.simverify.habit_cycle import SECTORS
from testbed.simverify.habit_cycle_audit import (
    _classification_metrics,
    _fit_episode_balanced_centroids,
    _fit_runtime_ready_centroids,
    _predict_centroids,
    _ready_feature,
    extract_candidate_features,
)

HELD_OUT_SOURCE_EPISODES = frozenset({1, 13, 25, 33})
CALIBRATION_SCHEMA = "simverify_habit_runtime_ready_calibration_v1"
DETECTOR_SCHEMA = "simverify_habit_runtime_ready_detector_v1"
TICK_SCHEMA = "simverify_habit_runtime_ready_tick_v1"


def build_habit_runtime_ready_calibration(
    *,
    repo_root: str | Path,
    definition_root: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    weights_path: str | Path,
    expected_weights_sha256: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Export train-fitted v11 visual centroids without reading held-out rows."""

    repository = Path(repo_root).resolve(strict=True)
    git = git_provenance(repository)
    if (
        not git.get("git_available")
        or git.get("branch") != "v2.0.0-simVerify"
        or bool(git.get("dirty"))
    ):
        raise ValueError("runtime-ready calibration requires a clean SimVerify tree")
    definition = Path(definition_root).resolve(strict=True)
    verification = verify_checksums(
        definition,
        definition / "checksums.sha256",
    )
    if not verification["ok"]:
        raise ValueError("definition checksum verification failed")
    decision = _read_json(definition / "definition_falsification_decision_v1.json")
    if decision.get("decision") != "accept":
        raise ValueError("runtime-ready calibration requires accepted definition")
    audit = _read_json(definition / "dig_ready_boundary_audit_v1.json")
    boundaries = _read_json(definition / "habit_cycle_boundaries_v1.json")
    rows = [
        row
        for row in boundaries["records"]
        if row.get("split") in {"train", "validation"}
        and row.get("dig_ready_reference_interval") is not None
        and row.get("hindsight_expert_target_sector") in SECTORS
    ]
    if not rows:
        raise ValueError("definition has no observable ready references")
    episode_ids = {int(row["episode_id"]) for row in rows}
    overlap = episode_ids & HELD_OUT_SOURCE_EPISODES
    if overlap:
        raise ValueError(f"held-out source episodes entered calibration: {sorted(overlap)}")
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("calibration requires non-empty train and validation rows")

    source = Path(source_root).resolve(strict=True)
    clean_dir = source / "clean_all_vds"
    if not clean_dir.is_dir():
        raise FileNotFoundError(clean_dir)
    paths = {
        episode_id: (clean_dir / f"episode_{episode_id}.hdf5").resolve(strict=True)
        for episode_id in sorted(episode_ids)
    }
    extractor = FrozenResNet18FeatureExtractor(
        weights_path,
        expected_checkpoint_sha256=expected_weights_sha256,
        device=device,
        batch_size=int(batch_size),
    )
    features = extract_candidate_features(
        rows,
        paths=paths,
        extractor=extractor,
    )
    ready_centroids = _fit_runtime_ready_centroids(train, features)
    sector_train_x = np.stack(
        [
            features[
                (
                    int(row["episode_id"]),
                    int(row["dig_ready_reference_interval"][1]) - 1,
                    "eye",
                )
            ]
            for row in train
        ]
    )
    sector_centroids = _fit_episode_balanced_centroids(
        sector_train_x,
        [str(row["hindsight_expert_target_sector"]) for row in train],
        [int(row["episode_id"]) for row in train],
    )
    if set(ready_centroids) != {"not_ready", "ready"}:
        raise ValueError("runtime-ready centroids do not contain both classes")
    if set(sector_centroids) != set(SECTORS):
        raise ValueError("sector centroids do not cover the frozen 3x1 labels")

    reproduction = _boundary_reproduction(rows, features, ready_centroids)
    validation_sector_expected = [
        str(row["hindsight_expert_target_sector"]) for row in validation
    ]
    validation_sector_x = np.stack(
        [
            features[
                (
                    int(row["episode_id"]),
                    int(row["dig_ready_reference_interval"][1]) - 1,
                    "eye",
                )
            ]
            for row in validation
        ]
    )
    validation_sector_prediction = _predict_centroids(
        validation_sector_x,
        sector_centroids,
    )
    validation_sector_metrics = _classification_metrics(
        validation_sector_expected,
        validation_sector_prediction,
    )
    expected_validation_rate = float(audit["causal_confirmation_validation_rate"])
    if not math.isclose(
        float(reproduction["validation"]["reference_match_rate"]),
        expected_validation_rate,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("exported runtime classifier does not reproduce v11")
    expected_sector = audit["visual_audit"]["ready_sector_eye_pair"]
    if (
        validation_sector_metrics["accuracy"] != expected_sector["accuracy"]
        or validation_sector_metrics["balanced_accuracy"]
        != expected_sector["balanced_accuracy"]
    ):
        raise ValueError("exported sector centroids do not reproduce v11")

    destination = Path(output_root).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable runtime-ready output exists: {destination}")
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    identities: list[dict[str, Any]] = []
    try:
        centroids_path = temporary / "runtime_ready_centroids_v1.npz"
        np.savez_compressed(
            centroids_path,
            ready=np.asarray(ready_centroids["ready"], dtype=np.float32),
            not_ready=np.asarray(ready_centroids["not_ready"], dtype=np.float32),
            sector_left=np.asarray(sector_centroids["left"], dtype=np.float32),
            sector_center=np.asarray(sector_centroids["center"], dtype=np.float32),
            sector_right=np.asarray(sector_centroids["right"], dtype=np.float32),
        )
        centroids_identity = artifact_identity(centroids_path)
        identities.append(centroids_identity)
        manifest_identity = write_json(
            temporary / "runtime_ready_manifest.json",
            {
                "schema": CALIBRATION_SCHEMA,
                "status": "complete",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git": git,
                "definition": {
                    "root": str(definition),
                    "audit_manifest_sha256": sha256_file(
                        definition / "audit_manifest.json"
                    ),
                    "boundary_artifact_sha256": sha256_file(
                        definition / "habit_cycle_boundaries_v1.json"
                    ),
                    "dig_ready_audit_sha256": sha256_file(
                        definition / "dig_ready_boundary_audit_v1.json"
                    ),
                },
                "classifier": {
                    "centroids_path": centroids_identity["path"],
                    "centroids_sha256": centroids_identity["sha256"],
                    "ready_feature": (
                        "l2(concat(l2(eye_left,eye_right),"
                        "l2(stick_down,stick_up)))"
                    ),
                    "ready_labels": ["not_ready", "ready"],
                    "sector_feature": "l2(eye_left,eye_right)",
                    "sector_labels": list(SECTORS),
                    "fit_scope": "train_source_episodes_only",
                    "fit_source_episode_ids": sorted(
                        {int(row["episode_id"]) for row in train}
                    ),
                },
                "numeric_contract": {
                    "ready_swing_speed_threshold": float(
                        audit["numeric_thresholds"]["ready"][
                            "swing_speed_threshold"
                        ]
                    ),
                    "ready_dwell_source_steps": int(
                        audit["causal_dwell_source_steps"]
                    ),
                    "dump_swing_threshold": float(
                        audit["numeric_thresholds"]["dump_release"][
                            "swing_threshold"
                        ]
                    ),
                    "sector_thresholds": audit["sector_thresholds"],
                    "arm_event": "first_abs_swing_qvel_above_ready_threshold_after_commit",
                    "candidate": (
                        "target_sector_and_abs_swing_qvel_lte_threshold_for_dwell"
                    ),
                },
                "validation": {
                    "boundary_reproduction": reproduction,
                    "ready_sector_eye_pair": validation_sector_metrics,
                },
                "feature_extractor": extractor.provenance,
                "observable_inputs": [
                    "eye_left_rgb",
                    "eye_right_rgb",
                    "stick_down_rgb",
                    "stick_up_rgb",
                    "swing_qpos",
                    "swing_qvel",
                    "scripted_target_sector",
                    "external_condition_lifecycle_state",
                ],
                "privilege_used": False,
                "future_observations_used": False,
                "held_out_source_episode_ids": sorted(HELD_OUT_SOURCE_EPISODES),
                "held_out_observation_read_count": 0,
                "evidence_scope": "recorded-observation/offline_runtime_calibration",
                "closed_loop_execution": False,
            },
        )
        identities.append(manifest_identity)
        checksums_identity = write_checksums(
            temporary,
            identities,
            path=temporary / "checksums.sha256",
        )
        os.replace(temporary, destination)
        written = verify_checksums(destination, destination / "checksums.sha256")
        if not written["ok"]:
            raise RuntimeError("runtime-ready checksum verification failed")
        return {
            "status": "completed",
            "output_root": str(destination),
            "manifest_sha256": manifest_identity["sha256"],
            "centroids_sha256": centroids_identity["sha256"],
            "checksums_sha256": checksums_identity["sha256"],
            "verification": written,
            "held_out_observation_read_count": 0,
        }
    finally:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()


class ObservableHabitReadyBoundaryDetector:
    """Causal v11 ready detector for one committed target."""

    def __init__(
        self,
        *,
        feature_extractor: Any,
        ready_centroids: Mapping[str, np.ndarray],
        sector_centroids: Mapping[str, np.ndarray],
        swing_speed_threshold: float,
        dwell_policy_ticks: int,
        dump_swing_threshold: float,
        sector_thresholds: Mapping[str, Any],
        artifact_provenance: Mapping[str, Any],
    ) -> None:
        if set(ready_centroids) != {"not_ready", "ready"}:
            raise ValueError("ready centroids must contain ready and not_ready")
        if set(sector_centroids) != set(SECTORS):
            raise ValueError("sector centroids must contain left, center, right")
        if int(dwell_policy_ticks) <= 0:
            raise ValueError("dwell_policy_ticks must be positive")
        self._extractor = feature_extractor
        self._ready_centroids = {
            key: _unit(value) for key, value in ready_centroids.items()
        }
        self._sector_centroids = {
            key: _unit(value) for key, value in sector_centroids.items()
        }
        self._swing_speed_threshold = float(swing_speed_threshold)
        self._dwell_policy_ticks = int(dwell_policy_ticks)
        self._dump_swing_threshold = float(dump_swing_threshold)
        self._sector_thresholds = json.loads(json.dumps(sector_thresholds))
        self._provenance = {
            "schema": DETECTOR_SCHEMA,
            "mode": "causal_live_adaptation_of_accepted_v11_ready_boundary",
            "artifact": dict(artifact_provenance),
            "feature_extractor": dict(feature_extractor.provenance),
            "privilege_used": False,
            "future_observations_used": False,
        }
        self.reset()

    @classmethod
    def from_calibration_artifacts(
        cls,
        *,
        calibration_root: str | Path,
        weights_path: str | Path,
        device: str = "cpu",
    ) -> ObservableHabitReadyBoundaryDetector:
        root = Path(calibration_root).resolve(strict=True)
        verification = verify_checksums(root, root / "checksums.sha256")
        if not verification["ok"]:
            raise ValueError("runtime-ready calibration checksum verification failed")
        manifest_path = root / "runtime_ready_manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("schema") != CALIBRATION_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("held_out_observation_read_count") != 0
            or manifest.get("privilege_used") is not False
        ):
            raise ValueError("runtime-ready calibration manifest is unsafe")
        centroids_path = root / "runtime_ready_centroids_v1.npz"
        if sha256_file(centroids_path) != manifest["classifier"]["centroids_sha256"]:
            raise ValueError("runtime-ready centroid SHA mismatch")
        with np.load(centroids_path, allow_pickle=False) as arrays:
            ready_centroids = {
                "ready": np.asarray(arrays["ready"], dtype=np.float32),
                "not_ready": np.asarray(arrays["not_ready"], dtype=np.float32),
            }
            sector_centroids = {
                sector: np.asarray(arrays[f"sector_{sector}"], dtype=np.float32)
                for sector in SECTORS
            }
        feature_contract = manifest["feature_extractor"]["checkpoint"]
        extractor = FrozenResNet18FeatureExtractor(
            weights_path,
            expected_checkpoint_sha256=str(feature_contract["sha256"]),
            device=device,
            batch_size=4,
        )
        numeric = manifest["numeric_contract"]
        source_dwell = int(numeric["ready_dwell_source_steps"])
        source_dt = 0.02
        policy_dt = 0.05
        dwell_policy_ticks = max(1, math.ceil(source_dwell * source_dt / policy_dt))
        return cls(
            feature_extractor=extractor,
            ready_centroids=ready_centroids,
            sector_centroids=sector_centroids,
            swing_speed_threshold=float(numeric["ready_swing_speed_threshold"]),
            dwell_policy_ticks=dwell_policy_ticks,
            dump_swing_threshold=float(numeric["dump_swing_threshold"]),
            sector_thresholds=numeric["sector_thresholds"],
            artifact_provenance={
                "root": str(root),
                "manifest_sha256": sha256_file(manifest_path),
                "centroids_sha256": sha256_file(centroids_path),
                "source_dwell_steps": source_dwell,
                "policy_dwell_ticks": dwell_policy_ticks,
            },
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._provenance))

    def reset(self) -> None:
        self._armed = False
        self._confirmed = False
        self._eligible_ticks = 0
        self._attempted_current_run = False

    def observe(
        self,
        *,
        policy_tick: int,
        observation: Mapping[str, Any],
        policy_observation: Mapping[str, Any],
        held_action: np.ndarray,
        condition_route: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        del held_action
        route = (
            None if condition_route is None else str(condition_route.get("route", ""))
        )
        condition = np.asarray(
            policy_observation.get("cycle_condition_v1", np.zeros(6)),
            dtype=np.float32,
        )
        target_sector = (
            None
            if condition.shape != (6,) or not np.isclose(condition[3:].sum(), 1.0)
            else SECTORS[int(np.argmax(condition[3:]))]
        )
        qpos = np.asarray(observation["qpos"], dtype=np.float32)
        qvel = np.asarray(observation["qvel"], dtype=np.float32)
        swing_qpos = float(qpos[0])
        swing_speed = abs(float(qvel[0]))
        return_active = bool(
            route == "next"
            and target_sector is not None
            and swing_speed > self._swing_speed_threshold
        )
        if not self._armed and not self._confirmed and return_active:
            self._armed = True

        qpos_sector, sector_confidence, boundary_review = classify_sector(
            swing_qpos,
            self._sector_thresholds,
        )
        eligible = bool(
            self._armed
            and not self._confirmed
            and route == "next"
            and target_sector is not None
            and qpos_sector == target_sector
            and swing_qpos < self._dump_swing_threshold
            and swing_speed <= self._swing_speed_threshold
        )
        if eligible:
            self._eligible_ticks += 1
        else:
            self._eligible_ticks = 0
            self._attempted_current_run = False
        candidate = bool(
            eligible
            and self._eligible_ticks >= self._dwell_policy_ticks
            and not self._attempted_current_run
        )
        ready_prediction = None
        visual_sector_prediction = None
        confirmed = False
        if candidate:
            self._attempted_current_run = True
            images = [
                np.asarray(policy_observation[f"image_{camera}"], dtype=np.uint8)
                for camera in ("video4", "video5", "video6", "video7")
            ]
            features = np.asarray(
                self._extractor.extract_rgb_batch(images),
                dtype=np.float32,
            )
            if features.shape != (4, 512):
                raise ValueError("runtime-ready feature shape must be (4,512)")
            eye = _unit(np.concatenate((features[0], features[1])))
            stick = _unit(np.concatenate((features[2], features[3])))
            ready = _unit(np.concatenate((eye, stick)))
            ready_prediction = _predict_one(ready, self._ready_centroids)
            visual_sector_prediction = _predict_one(eye, self._sector_centroids)
            confirmed = bool(
                ready_prediction == "ready"
                and visual_sector_prediction == target_sector
            )
            if confirmed:
                self._confirmed = True
        return {
            "schema": TICK_SCHEMA,
            "policy_tick": int(policy_tick),
            "state": (
                "confirmed"
                if self._confirmed
                else "armed"
                if self._armed
                else "searching_return_activation"
            ),
            "confirmed": confirmed,
            "condition_route_before_predict": route,
            "target_sector": target_sector,
            "swing_qpos": swing_qpos,
            "abs_swing_qvel": swing_speed,
            "return_active": return_active,
            "qpos_sector": qpos_sector,
            "qpos_sector_confidence": float(sector_confidence),
            "qpos_sector_boundary_review": bool(boundary_review),
            "eligible": eligible,
            "eligible_ticks": int(self._eligible_ticks),
            "candidate": candidate,
            "ready_visual_prediction": ready_prediction,
            "visual_sector_prediction": visual_sector_prediction,
        }


def _boundary_reproduction(
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[int, int, str], np.ndarray],
    centroids: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        matches = 0
        confirmed = 0
        for row in selected:
            predicted_step = None
            for step in row["numeric_causal_candidate_steps"]:
                feature = _ready_feature(row, int(step), features)
                if _predict_centroids(feature[None], centroids)[0] == "ready":
                    predicted_step = int(step)
                    break
            if predicted_step is not None:
                confirmed += 1
                start, end = map(int, row["dig_ready_reference_interval"])
                matches += int(start <= predicted_step < end)
        result[split] = {
            "candidate_count": len(selected),
            "confirmed_count": confirmed,
            "reference_match_count": matches,
            "reference_match_rate": float(matches / max(1, len(selected))),
        }
    return result


def _predict_one(
    feature: np.ndarray,
    centroids: Mapping[str, np.ndarray],
) -> str:
    labels = sorted(centroids)
    scores = [_unit(feature) @ _unit(centroids[label]) for label in labels]
    return labels[int(np.argmax(scores))]


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("feature vector must have finite non-zero norm")
    return array / norm


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
