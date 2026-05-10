from __future__ import annotations

import unittest

import numpy as np

from testbed.actions.smoothing import ActionResponseSmoother, AxisResponseProfile


class ActionSmoothingTests(unittest.TestCase):
    def test_axis_response_profile_deadzone_and_exponent(self) -> None:
        profile = AxisResponseProfile(deadzone=0.1, exponent=2.0)
        self.assertEqual(profile.remap(0.05), 0.0)
        self.assertAlmostEqual(profile.remap(-0.05), 0.0)
        self.assertAlmostEqual(profile.remap(1.0), 1.0)
        self.assertAlmostEqual(profile.remap(-1.0), -1.0)

        remapped = profile.remap(0.55)
        expected = ((0.55 - 0.1) / (1.0 - 0.1)) ** 2.0
        self.assertAlmostEqual(remapped, expected)

    def test_axis_response_profile_uses_attack_release_and_recenter_rates(self) -> None:
        profile = AxisResponseProfile(
            deadzone=0.0,
            attack_rate=2.0,
            release_rate=3.0,
            recenter_rate=4.0,
            exponent=1.0,
        )
        self.assertAlmostEqual(profile.apply(raw_value=1.0, current_value=0.0, delta_time=0.1), 0.2)
        self.assertAlmostEqual(profile.apply(raw_value=0.5, current_value=0.8, delta_time=0.1), 0.5)
        self.assertAlmostEqual(profile.apply(raw_value=0.0, current_value=0.8, delta_time=0.1), 0.4)

    def test_action_response_smoother_tracks_state_per_axis(self) -> None:
        smoother = ActionResponseSmoother(
            deadzone=0.0,
            attack_rate=2.0,
            release_rate=2.0,
            recenter_rate=2.0,
            exponent=1.0,
            default_dt=0.1,
        )
        first = smoother.apply(np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32), delta_time=0.1)
        np.testing.assert_allclose(first, np.array([0.2, 0.0, -0.2, 0.2], dtype=np.float32))

        second = smoother.apply(np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32), delta_time=0.1)
        np.testing.assert_allclose(second, np.array([0.4, 0.0, -0.4, 0.4], dtype=np.float32))

        smoother.reset()
        reset_step = smoother.apply(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), delta_time=0.1)
        np.testing.assert_allclose(reset_step, np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
