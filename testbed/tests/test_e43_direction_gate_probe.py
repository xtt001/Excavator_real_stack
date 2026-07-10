import numpy as np

from scripts.e43_direction_gate_probe import apply_direction_probability_gate


def test_apply_direction_probability_gate_scales_inactive_policy_directions() -> None:
    policy = np.asarray([[0.8, -0.6, 0.4, -0.2]], dtype=np.float32)
    prob = np.asarray([[0.9, 0.1, 0.2, 0.8, 0.1, 0.9, 0.7, 0.1]], dtype=np.float32)

    gated = apply_direction_probability_gate(policy, prob, threshold=0.75, inactive_scale=0.25)

    np.testing.assert_allclose(gated, [[0.8, -0.6, 0.1, -0.05]], rtol=1e-6)
