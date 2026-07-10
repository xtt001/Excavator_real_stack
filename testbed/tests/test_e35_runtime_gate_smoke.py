import numpy as np

from scripts.e35_runtime_gate_smoke import (
    active_mask_from_gohome_gate,
    json_safe,
    parse_gohome_gate_name,
    parse_phase_gate_name,
)


def test_parse_phase_gate_name_reads_simple_threshold_and_inactive_scale() -> None:
    parsed = parse_phase_gate_name("simple_0.15_s0.50")

    assert parsed == {"mode": "simple", "threshold": 0.15, "inactive_scale": 0.5}


def test_parse_gohome_gate_name_reads_two_stage_thresholds() -> None:
    parsed = parse_gohome_gate_name("learned_tail_t0.97_tc10_e0.80_ec3")

    assert parsed == {
        "candidate_threshold": 0.97,
        "candidate_consecutive_steps": 10,
        "eligibility_threshold": 0.80,
        "eligibility_consecutive_steps": 3,
    }


def test_active_mask_from_gohome_gate_applies_candidate_before_eligibility() -> None:
    candidate_prob = np.asarray([0.1, 0.99, 0.99, 0.99, 0.1], dtype=np.float32)
    eligibility_prob = np.asarray([0.99, 0.99, 0.99, 0.1, 0.99], dtype=np.float32)
    gate = {
        "candidate_threshold": 0.9,
        "candidate_consecutive_steps": 2,
        "eligibility_threshold": 0.9,
        "eligibility_consecutive_steps": 2,
    }

    active = active_mask_from_gohome_gate(candidate_prob, eligibility_prob, gate)

    np.testing.assert_array_equal(active, [False, False, True, False, False])


def test_json_safe_converts_numpy_values() -> None:
    payload = {
        "array": np.asarray([1.0, 2.0], dtype=np.float32),
        "scalar": np.float32(3.5),
        "nested": {"flag": np.bool_(True)},
    }

    assert json_safe(payload) == {"array": [1.0, 2.0], "scalar": 3.5, "nested": {"flag": True}}
