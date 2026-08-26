"""Goal-only planner for composing a conditioned ACT policy.

The planner owns the order of target sides.  ACT remains the only component
that produces actuator commands.  Each committed goal is exposed to the
policy as ``real_transition_condition_v1=[target_side_code, 1]``; no planner
template, cycle index, or future target is placed in the policy input.

This module is intentionally hardware agnostic.  It can be used to generate
an auditable schedule for a runner or to attach the current condition to an
observation before calling :class:`PolicyActionSource`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SIDE_CODES = {"A": -1, "B": 1}
CONDITION_KEY = "real_transition_condition_v1"
PLANNER_SCHEMA = "act_cycle_planner_v1"
SCRIPT_SCHEMA = "act_cycle_script_v1"


class CyclePlannerError(ValueError):
    """Raised when a cycle sequence or boundary transition is invalid."""


def parse_side_pattern(value: str) -> tuple[str, ...]:
    """Parse a compact full-side pattern such as ``ABBABABA``.

    The first character is the initial ready side.  Consecutive equal sides
    are valid and represent an in-place target (for example ``B->B``).
    Separators are accepted for operator convenience and ignored.
    """

    text = "".join(str(value or "").upper().split())
    text = text.replace("-", "").replace(">", "").replace(",", "")
    if len(text) < 2:
        raise CyclePlannerError(
            "side pattern must contain an initial side and at least one target"
        )
    invalid = sorted(set(text) - set(SIDE_CODES))
    if invalid:
        raise CyclePlannerError(
            f"side pattern contains invalid side(s): {', '.join(invalid)}"
        )
    return tuple(text)


@dataclass(frozen=True)
class PlannedCycle:
    """One goal transition exposed by the planner."""

    cycle_index: int
    goal_epoch: int
    current_side: str
    target_side: str
    target_side_code: int
    transition: str
    script_step_index: int | None = None
    step_id: str | None = None
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def condition(self) -> tuple[float, float]:
        return (float(self.target_side_code), 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "goal_epoch": int(self.goal_epoch),
            "current_side": self.current_side,
            "target_side": self.target_side,
            "target_side_code": int(self.target_side_code),
            "transition": self.transition,
            CONDITION_KEY: list(self.condition),
            **(
                {"script_step_index": int(self.script_step_index)}
                if self.script_step_index is not None
                else {}
            ),
            **({"step_id": self.step_id} if self.step_id else {}),
            **({"label": self.label} if self.label else {}),
            **({"metadata": dict(self.metadata)} if self.metadata else {}),
        }


@dataclass(frozen=True)
class ScriptStep:
    """One arbitrary target in a script-defined cycle plan."""

    target_side: str
    step_id: str
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "step_id": self.step_id,
            "target_side": self.target_side,
        }
        if self.label:
            result["label"] = self.label
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


class ABCyclePlanner:
    """Stateful, goal-only sequencer for a repeated A/B excavation pattern.

    ``pattern`` is the complete side path, including the initial ready side.
    For ``ABBABABA`` the first pass is ``A->B, B->B, B->A, A->B, ...``.
    When ``loop=True``, the path wraps from the last pattern side to the first
    side, so a repeated ``ABBABABA`` pass adds ``A->A`` at the boundary.  A
    caller may set ``max_cycles`` to stop after a finite number of goals.

    ``commit_goal`` and ``mark_target_ready`` are separate on purpose.  The
    planner cannot advance merely because a goal was requested; the observed
    ready boundary must confirm the expected target side first.
    """

    def __init__(
        self,
        pattern: str | tuple[str, ...] = "ABBABABA",
        *,
        loop: bool = True,
        max_cycles: int | None = None,
    ) -> None:
        self.pattern = (
            parse_side_pattern(pattern)
            if isinstance(pattern, str)
            else _validate_pattern(pattern)
        )
        self.loop = bool(loop)
        if max_cycles is not None and int(max_cycles) <= 0:
            raise CyclePlannerError("max_cycles must be positive when provided")
        self.max_cycles = None if max_cycles is None else int(max_cycles)
        self.reset()

    def reset(self) -> None:
        self._cycle_index = 0
        self._goal_epoch = 0
        self._committed: PlannedCycle | None = None
        self._completed = False

    @property
    def cycle_index(self) -> int:
        return int(self._cycle_index)

    @property
    def goal_epoch(self) -> int:
        return int(self._goal_epoch)

    @property
    def initial_side(self) -> str:
        return self.pattern[0]

    @property
    def done(self) -> bool:
        return bool(self._completed)

    @property
    def committed_goal(self) -> PlannedCycle | None:
        return self._committed

    def peek_goal(self) -> PlannedCycle | None:
        """Return the next goal without committing it."""

        if self.done:
            return None
        if self.max_cycles is not None and self._cycle_index >= self.max_cycles:
            return None
        current = (
            self.initial_side
            if self._cycle_index == 0
            else self._target_for_cycle_index(self._cycle_index - 1)
        )
        try:
            target = self._target_for_cycle_index(self._cycle_index)
        except CyclePlannerError:
            return None
        return PlannedCycle(
            cycle_index=int(self._cycle_index),
            goal_epoch=int(self._goal_epoch + 1),
            current_side=current,
            target_side=target,
            target_side_code=int(SIDE_CODES[target]),
            transition=f"{current}->{target}",
        )

    def commit_goal(self) -> PlannedCycle:
        """Commit the next goal and return its condition payload."""

        if self._committed is not None:
            raise CyclePlannerError(
                "current goal is already committed; wait for target-ready"
            )
        goal = self.peek_goal()
        if goal is None:
            self._completed = True
            raise CyclePlannerError("cycle plan has no remaining goal")
        self._goal_epoch = int(goal.goal_epoch)
        self._committed = goal
        return goal

    def mark_target_ready(self, realized_side: str) -> PlannedCycle | None:
        """Close the committed cycle after an observed ready boundary.

        A mismatch leaves the planner committed and raises, allowing the
        caller to stop safely and record the failed boundary.
        """

        if self._committed is None:
            raise CyclePlannerError("target-ready received before goal commit")
        actual = str(realized_side or "").upper()
        if actual not in SIDE_CODES:
            raise CyclePlannerError("realized_side must be A or B")
        if actual != self._committed.target_side:
            raise CyclePlannerError(
                "realized target side does not match committed goal: "
                f"expected {self._committed.target_side}, got {actual}"
            )
        self._cycle_index += 1
        self._committed = None
        next_goal = self.peek_goal()
        if next_goal is None:
            self._completed = True
        return next_goal

    def condition(self) -> np.ndarray:
        """Return the currently committed condition as a float32 vector."""

        if self._committed is None:
            raise CyclePlannerError("no committed goal is available")
        return np.asarray(self._committed.condition, dtype=np.float32)

    def apply_condition(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Copy an observation and attach only the current ACT condition."""

        if self._committed is None:
            raise CyclePlannerError("cannot apply condition before goal commit")
        result = dict(observation)
        result[CONDITION_KEY] = self.condition().copy()
        return result

    def snapshot_state(self) -> dict[str, Any]:
        """Return a serialisable state for action-source branch replay."""

        return {
            "cycle_index": int(self._cycle_index),
            "goal_epoch": int(self._goal_epoch),
            "completed": bool(self._completed),
            "committed": (
                None
                if self._committed is None
                else self._committed.as_dict()
            ),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`snapshot_state`."""

        if not isinstance(state, Mapping):
            raise CyclePlannerError("planner state must be a mapping")
        cycle_index = int(state.get("cycle_index", -1))
        goal_epoch = int(state.get("goal_epoch", -1))
        if cycle_index < 0 or goal_epoch < 0:
            raise CyclePlannerError("planner state indices must be non-negative")
        committed_raw = state.get("committed")
        committed: PlannedCycle | None = None
        if committed_raw is not None:
            if not isinstance(committed_raw, Mapping):
                raise CyclePlannerError("planner committed state must be a mapping")
            try:
                committed = PlannedCycle(
                    cycle_index=int(committed_raw["cycle_index"]),
                    goal_epoch=int(committed_raw["goal_epoch"]),
                    current_side=str(committed_raw["current_side"]),
                    target_side=str(committed_raw["target_side"]),
                    target_side_code=int(committed_raw["target_side_code"]),
                    transition=str(committed_raw["transition"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CyclePlannerError(
                    "planner committed state is malformed"
                ) from exc
            if committed.target_side_code != SIDE_CODES.get(
                committed.target_side
            ):
                raise CyclePlannerError(
                    "planner committed target code disagrees with target side"
                )
        self._cycle_index = cycle_index
        self._goal_epoch = goal_epoch
        self._committed = committed
        self._completed = bool(state.get("completed", False))

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-safe frozen schedule for audit or dry-run use."""

        goals: list[dict[str, Any]] = []
        count = self.max_cycles
        if count is None:
            count = len(self.pattern) - 1
        for index in range(int(count)):
            current = (
                self.initial_side
                if index == 0
                else self._target_for_cycle_index(index - 1)
            )
            try:
                target = self._target_for_cycle_index(index)
            except CyclePlannerError:
                break
            goal = PlannedCycle(
                cycle_index=index,
                goal_epoch=index + 1,
                current_side=current,
                target_side=target,
                target_side_code=SIDE_CODES[target],
                transition=f"{current}->{target}",
            )
            goals.append(goal.as_dict())
        return {
            "schema": PLANNER_SCHEMA,
            "condition_schema": CONDITION_KEY,
            "pattern": "".join(self.pattern),
            "initial_side": self.initial_side,
            "loop": self.loop,
            "max_cycles": self.max_cycles,
            "cycles": goals,
            "policy_input_boundary": [CONDITION_KEY],
            "action_owner": "ACT",
        }

    def write_manifest(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write a planner manifest, refusing accidental overwrite by default."""

        output = Path(path)
        if output.exists() and not overwrite:
            raise FileExistsError(f"planner manifest already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.manifest(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    def _target_for_cycle_index(self, index: int) -> str:
        if index < 0:
            raise CyclePlannerError("cycle index must be non-negative")
        pattern_index = index + 1
        if pattern_index >= len(self.pattern) and not self.loop:
            raise CyclePlannerError("non-looping cycle plan is complete")
        return self.pattern[pattern_index % len(self.pattern)]


class ScriptCyclePlanner(ABCyclePlanner):
    """Variable-length, variable-order goal script planner.

    Unlike the compact ``ABCyclePlanner`` shorthand, a script stores the
    initial ready side and an explicit list of target steps.  A looping script
    repeats that target list directly; it does not invent an extra transition
    from the last target back to the initial side.  This lets a caller express
    arbitrary schedules such as ``A -> B -> B -> A -> A -> B`` and attach
    per-step labels/metadata for logging without leaking them to ACT.
    """

    def __init__(
        self,
        *,
        initial_side: str,
        steps: tuple[Any, ...] | list[Any],
        loop: bool = False,
        max_cycles: int | None = None,
        script_id: str = "",
        source_path: str | None = None,
    ) -> None:
        self.script_id = str(script_id or "")
        self.source_path = None if source_path is None else str(source_path)
        self.steps = tuple(_coerce_script_step(step, index) for index, step in enumerate(steps))
        if not self.steps:
            raise CyclePlannerError("cycle script must contain at least one step")
        initial = str(initial_side or "").upper()
        if initial not in SIDE_CODES:
            raise CyclePlannerError("cycle script initial_side must be A or B")
        # Keep the base fields for shared status/snapshot behaviour.  The
        # overridden target/peek/manifest methods below use explicit steps.
        self.pattern = (initial, *(step.target_side for step in self.steps))
        self.loop = bool(loop)
        if max_cycles is not None and int(max_cycles) <= 0:
            raise CyclePlannerError("max_cycles must be positive when provided")
        self.max_cycles = None if max_cycles is None else int(max_cycles)
        self.reset()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        loop: bool | None = None,
        max_cycles: int | None = None,
        source_path: str | None = None,
    ) -> ScriptCyclePlanner:
        if not isinstance(value, Mapping):
            raise CyclePlannerError("cycle script must be a mapping")
        payload = dict(value)
        schema = payload.get("schema")
        if schema not in (None, SCRIPT_SCHEMA):
            raise CyclePlannerError(
                f"unsupported cycle script schema {schema!r}; expected {SCRIPT_SCHEMA!r}"
            )
        initial = payload.get("initial_side", payload.get("start_side"))
        if initial is None:
            raise CyclePlannerError("cycle script requires initial_side")
        raw_steps = payload.get("steps", payload.get("targets"))
        if not isinstance(raw_steps, (list, tuple)):
            raise CyclePlannerError("cycle script requires a steps list")
        resolved_loop = bool(payload.get("loop", False)) if loop is None else bool(loop)
        resolved_max = payload.get("max_cycles") if max_cycles is None else max_cycles
        return cls(
            initial_side=str(initial),
            steps=tuple(
                _coerce_script_step(step, index)
                for index, step in enumerate(raw_steps)
            ),
            loop=resolved_loop,
            max_cycles=(
                None if resolved_max is None else int(resolved_max)
            ),
            script_id=str(payload.get("script_id", payload.get("id", "")) or ""),
            source_path=source_path,
        )

    @classmethod
    def from_script(
        cls,
        path: str | Path,
        *,
        loop: bool | None = None,
        max_cycles: int | None = None,
    ) -> ScriptCyclePlanner:
        script_path = Path(path)
        if not script_path.is_file():
            raise FileNotFoundError(f"cycle script does not exist: {script_path}")
        payload = _load_cycle_script_payload(script_path)
        return cls.from_mapping(
            payload,
            loop=loop,
            max_cycles=max_cycles,
            source_path=str(script_path),
        )

    def peek_goal(self) -> PlannedCycle | None:
        if self.done:
            return None
        if self.max_cycles is not None and self._cycle_index >= self.max_cycles:
            return None
        try:
            target = self._target_for_cycle_index(self._cycle_index)
        except CyclePlannerError:
            return None
        current = (
            self.initial_side
            if self._cycle_index == 0
            else self._target_for_cycle_index(self._cycle_index - 1)
        )
        step = self.steps[self._cycle_index % len(self.steps)]
        return PlannedCycle(
            cycle_index=int(self._cycle_index),
            goal_epoch=int(self._goal_epoch + 1),
            current_side=current,
            target_side=target,
            target_side_code=int(SIDE_CODES[target]),
            transition=f"{current}->{target}",
            script_step_index=int(self._cycle_index % len(self.steps)),
            step_id=step.step_id,
            label=step.label,
            metadata=dict(step.metadata),
        )

    def manifest(self) -> dict[str, Any]:
        count = self.max_cycles
        if count is None:
            count = len(self.steps)
        cycles: list[dict[str, Any]] = []
        for index in range(int(count)):
            try:
                goal = self._goal_for_index(index)
            except CyclePlannerError:
                break
            cycles.append(goal.as_dict())
        return {
            "schema": PLANNER_SCHEMA,
            "condition_schema": CONDITION_KEY,
            "planner_type": "script",
            "script": {
                "schema": SCRIPT_SCHEMA,
                "script_id": self.script_id,
                "source_path": self.source_path,
                "initial_side": self.initial_side,
                "steps": [step.as_dict() for step in self.steps],
                "loop": self.loop,
                "max_cycles": self.max_cycles,
            },
            "pattern": "".join(self.pattern),
            "initial_side": self.initial_side,
            "loop": self.loop,
            "max_cycles": self.max_cycles,
            "cycles": cycles,
            "policy_input_boundary": [CONDITION_KEY],
            "action_owner": "ACT",
        }

    def _target_for_cycle_index(self, index: int) -> str:
        if index < 0:
            raise CyclePlannerError("cycle index must be non-negative")
        if index >= len(self.steps) and not self.loop:
            raise CyclePlannerError("non-looping cycle script is complete")
        return self.steps[index % len(self.steps)].target_side

    def _goal_for_index(self, index: int) -> PlannedCycle:
        target = self._target_for_cycle_index(index)
        current = (
            self.initial_side
            if index == 0
            else self._target_for_cycle_index(index - 1)
        )
        step_index = int(index % len(self.steps))
        step = self.steps[step_index]
        return PlannedCycle(
            cycle_index=int(index),
            goal_epoch=int(index + 1),
            current_side=current,
            target_side=target,
            target_side_code=int(SIDE_CODES[target]),
            transition=f"{current}->{target}",
            script_step_index=step_index,
            step_id=step.step_id,
            label=step.label,
            metadata=dict(step.metadata),
        )


def _coerce_script_step(value: Any, index: int) -> ScriptStep:
    if isinstance(value, ScriptStep):
        target = value.target_side
        payload = {
            "step_id": value.step_id,
            "label": value.label,
            "metadata": value.metadata,
        }
    elif isinstance(value, str):
        target = value
        payload: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        payload = dict(value)
        target = payload.get("target_side", payload.get("target"))
    else:
        raise CyclePlannerError(
            f"cycle script step {index} must be a side string or mapping"
        )
    target_side = str(target or "").upper()
    if target_side not in SIDE_CODES:
        raise CyclePlannerError(
            f"cycle script step {index} target_side must be A or B"
        )
    step_id = str(payload.get("step_id", payload.get("id", f"step_{index:03d}")))
    if not step_id.strip():
        raise CyclePlannerError(f"cycle script step {index} step_id must not be empty")
    label = str(payload.get("label", "") or "")
    metadata_raw = payload.get("metadata", {}) or {}
    if not isinstance(metadata_raw, Mapping):
        raise CyclePlannerError(
            f"cycle script step {index} metadata must be a mapping"
        )
    metadata = dict(metadata_raw)
    try:
        json.dumps(metadata, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CyclePlannerError(
            f"cycle script step {index} metadata must be JSON serialisable"
        ) from exc
    return ScriptStep(
        target_side=target_side,
        step_id=step_id,
        label=label,
        metadata=metadata,
    )


def _load_cycle_script_payload(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CyclePlannerError(f"cannot read cycle script {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CyclePlannerError("cycle script root must be a mapping")
    nested = payload.get("planner")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise CyclePlannerError("cycle script planner field must be a mapping")
        payload = nested
    return payload


def _validate_pattern(value: tuple[str, ...]) -> tuple[str, ...]:
    if len(value) < 2:
        raise CyclePlannerError(
            "side pattern must contain an initial side and at least one target"
        )
    result = tuple(str(item).upper() for item in value)
    invalid = sorted(set(result) - set(SIDE_CODES))
    if invalid:
        raise CyclePlannerError(
            f"side pattern contains invalid side(s): {', '.join(invalid)}"
        )
    return result
