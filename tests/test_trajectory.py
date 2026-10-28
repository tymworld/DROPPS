import unittest
from types import SimpleNamespace

from dropps.share.trajectory import trajectory_class


class _Frame:
    def __init__(self, time):
        self.time = time


def _trajectory(frame_times_ps):
    trajectory = trajectory_class.__new__(trajectory_class)
    trajectory.Universe = SimpleNamespace(
        trajectory=[_Frame(time) for time in frame_times_ps]
    )
    return trajectory


class TrajectoryTimeSelectionTests(unittest.TestCase):
    def test_bounds_are_selected_from_actual_frame_times(self):
        trajectory = _trajectory([0.0, 1000.0, 2000.0, 3000.0])

        self.assertEqual(trajectory.time2frame(1.5, 2.5), (2, 2, 1))

    def test_requested_interval_rounds_to_the_nearest_frame_stride(self):
        trajectory = _trajectory([0.0, 1000.0, 2000.0, 3000.0])

        self.assertEqual(trajectory.time2frame(0.0, 3.0, 1.5), (0, 3, 2))

    def test_nonuniform_times_reject_a_constant_physical_interval(self):
        trajectory = _trajectory([0.0, 900.0, 2100.0, 3000.0])

        with self.assertRaisesRegex(ValueError, "nonuniform"):
            trajectory.time2frame(0.0, 3.0, 1.0)

    def test_nonpositive_interval_is_rejected(self):
        trajectory = _trajectory([0.0, 1000.0])

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            trajectory.time2frame(0.0, 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
