"""Task-level contracts for expert-habit scripted conditioned cycles.

This module deliberately owns semantics only.  It does not train a planner,
read simulator privilege, or turn hindsight outcomes into recorded commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SECTORS = ("left", "center", "right")
RELATIVE_INTENTS = ("stay", "step_left", "step_right")
DIAGNOSTIC_NONADJACENT = "nonadjacent_jump"


def relative_intent(current_sector: str, target_sector: str) -> str:
    """Return the expert-habit transition class for one sector pair."""

    current = _sector_index(current_sector)
    target = _sector_index(target_sector)
    delta = target - current
    if delta == 0:
        return "stay"
    if delta == -1:
        return "step_left"
    if delta == 1:
        return "step_right"
    return DIAGNOSTIC_NONADJACENT


def resolve_target_sector(current_sector: str, intent: str) -> str:
    """Resolve a supported relative intent and fail closed at work-area edges."""

    current = _sector_index(current_sector)
    if intent not in RELATIVE_INTENTS:
        raise ValueError(f"unsupported relative intent: {intent!r}")
    delta = {"stay": 0, "step_left": -1, "step_right": 1}[intent]
    target = current + delta
    if target < 0 or target >= len(SECTORS):
        raise ValueError(
            f"relative intent {intent!r} leaves 3x1 work area from "
            f"{current_sector!r}"
        )
    return SECTORS[target]


def cycle_action_valid_mask(
    *,
    observation_tick: int,
    cycle_end_tick: int,
    horizon: int,
) -> np.ndarray:
    """Mask action targets that belong to the next half-open cycle."""

    start = int(observation_tick)
    end = int(cycle_end_tick)
    size = int(horizon)
    if start < 0:
        raise ValueError("observation_tick must be non-negative")
    if end <= start:
        raise ValueError("cycle_end_tick must be greater than observation_tick")
    if size <= 0:
        raise ValueError("horizon must be positive")
    return (start + np.arange(size, dtype=np.int64)) < end


@dataclass
class HabitCycleLifecycle:
    """Fail-closed lifecycle for one scripted ready-to-ready attempt."""

    cycle_id: int
    current_sector: str
    condition_source: str = "scripted_fixed_scenario"
    _events: list[str] = field(default_factory=list, init=False, repr=False)
    _committed_target_sector: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _relative_intent: str | None = field(default=None, init=False, repr=False)
    _stopped: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.cycle_id) < 0:
            raise ValueError("cycle_id must be non-negative")
        _sector_index(self.current_sector)

    @property
    def ready_detection_armed(self) -> bool:
        return self._events == [
            "leave_initial_ready",
            "dig_entry",
            "carry",
            "dump_start",
            "dump_end",
        ]

    @property
    def committed_target_sector(self) -> str | None:
        return self._committed_target_sector

    def observe_event(self, event: str) -> None:
        """Advance the observable lifecycle in its only valid order."""

        if self._stopped:
            raise RuntimeError("cycle lifecycle is already stopped")
        order = [
            "leave_initial_ready",
            "dig_entry",
            "carry",
            "dump_start",
            "dump_end",
        ]
        if len(self._events) >= len(order):
            raise ValueError("all required observable events are already present")
        expected = order[len(self._events)]
        if event != expected:
            raise ValueError(
                f"invalid observable event order: expected {expected!r}, "
                f"got {event!r}"
            )
        self._events.append(event)

    def commit_after_dump(self, intent: str) -> dict[str, Any]:
        """Commit the fixed-script target only after a complete dump sequence."""

        if not self.ready_detection_armed:
            raise RuntimeError("target may be committed only after dump_end")
        if self._committed_target_sector is not None:
            raise RuntimeError("target is already committed")
        target = resolve_target_sector(self.current_sector, intent)
        self._relative_intent = intent
        self._committed_target_sector = target
        return {
            "cycle_id": int(self.cycle_id),
            "current_sector": self.current_sector,
            "relative_intent": intent,
            "scripted_target_sector": target,
            "condition_source": self.condition_source,
        }

    def confirm_target_ready(
        self,
        realized_target_sector: str,
        *,
        physical_effect_validated: bool | None = None,
    ) -> dict[str, Any]:
        """Complete only after ordered events and an exact committed target."""

        if self._stopped:
            raise RuntimeError("cycle lifecycle is already stopped")
        if not self.ready_detection_armed:
            raise RuntimeError("target-ready detector is not armed")
        if self._committed_target_sector is None or self._relative_intent is None:
            raise RuntimeError("missing committed target after dump_end")
        _sector_index(realized_target_sector)
        completed = realized_target_sector == self._committed_target_sector
        self._stopped = True
        return {
            "cycle_id": int(self.cycle_id),
            "current_sector": self.current_sector,
            "relative_intent": self._relative_intent,
            "scripted_target_sector": self._committed_target_sector,
            "hindsight_expert_target_sector": None,
            "realized_target_sector": realized_target_sector,
            "observable_cycle_completed": bool(completed),
            "physical_effect_validated": physical_effect_validated,
            "condition_source": self.condition_source,
            "terminal_reason": (
                "target_dig_ready_confirmed"
                if completed
                else "wrong_sector_dig_ready_confirmed"
            ),
        }

    def stop_without_completion(self, reason: str) -> dict[str, Any]:
        """Stop an attempt without inventing a successful shared boundary."""

        if self._stopped:
            raise RuntimeError("cycle lifecycle is already stopped")
        if not str(reason).strip():
            raise ValueError("stop reason must be non-empty")
        self._stopped = True
        return {
            "cycle_id": int(self.cycle_id),
            "current_sector": self.current_sector,
            "relative_intent": self._relative_intent,
            "scripted_target_sector": self._committed_target_sector,
            "hindsight_expert_target_sector": None,
            "realized_target_sector": None,
            "observable_cycle_completed": False,
            "physical_effect_validated": None,
            "condition_source": self.condition_source,
            "terminal_reason": str(reason),
        }


def _sector_index(sector: str) -> int:
    try:
        return SECTORS.index(str(sector))
    except ValueError as exc:
        raise ValueError(f"invalid 3x1 sector: {sector!r}") from exc
