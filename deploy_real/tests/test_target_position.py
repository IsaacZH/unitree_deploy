import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R


DEPLOY_REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEPLOY_REAL_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_REAL_DIR)

from common.navigation_command_manager import NavigationCommandManager
from terms import target_position


LAB_SAMPLES = [
    {
        "robot_position_world": [46.69829559326172, 13.06136703491211, 0.3287915289402008],
        "goal_position_world": [49.400001525878906, 5.700000762939453, 0.5696055293083191],
        "computed_target_position": [0.9934558272361755, -0.10898947715759277, 0.03415911644697189, 2.1798720359802246],
    },
    {
        "robot_position_world": [47.88174057006836, 9.061556816101074, 0.3281102776527405],
        "goal_position_world": [49.400001525878906, 5.700000762939453, 0.5696055293083191],
        "computed_target_position": [0.9886523485183716, -0.1325116753578186, 0.07076102495193481, 1.5468000173568726],
    },
    {
        "robot_position_world": [49.39496612548828, 5.739858627319336, 0.3243178129196167],
        "goal_position_world": [49.400001525878906, 5.700000762939453, 0.5696055293083191],
        "computed_target_position": [0.12586519122123718, -0.07359146326780319, 0.9893140196800232, 0.22198626399040222],
    },
    {
        "robot_position_world": [57.02446746826172, 3.0036866664886475, 0.3280639946460724],
        "goal_position_world": [48.900001525878906, 15.0, 0.2306550145149231],
        "computed_target_position": [0.9949220418930054, 0.09997330605983734, 0.011641215533018112, 2.7401225566864014],
    },
]


def _estimate_quaternion_wxyz(goal_w, robot_w, expected_dir_b):
    # Lab logs do not include quaternion. Build one minimal rotation that aligns
    # expected body-frame direction to the world-frame goal direction.
    world_dir = np.asarray(goal_w, dtype=np.float64) - np.asarray(robot_w, dtype=np.float64)
    world_dir = world_dir / np.linalg.norm(world_dir)
    body_dir = np.asarray(expected_dir_b, dtype=np.float64)
    body_dir = body_dir / np.linalg.norm(body_dir)

    rot_world_from_body, _ = R.align_vectors(
        world_dir.reshape(1, 3),
        body_dir.reshape(1, 3),
    )
    quat_xyzw = rot_world_from_body.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


class TestTargetPositionFromLabSamples(unittest.TestCase):
    def test_compute_goal_command_body_matches_lab_samples(self):
        for sample in LAB_SAMPLES:
            robot_w = np.asarray(sample["robot_position_world"], dtype=np.float32)
            goal_w = np.asarray(sample["goal_position_world"], dtype=np.float32)
            expected = np.asarray(sample["computed_target_position"], dtype=np.float32)

            quat_wxyz = _estimate_quaternion_wxyz(goal_w, robot_w, expected[:3])
            got, _ = NavigationCommandManager.compute_goal_command_body(goal_w, robot_w, quat_wxyz)

            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_target_position_term_matches_lab_samples(self):
        for sample in LAB_SAMPLES:
            robot_w = np.asarray(sample["robot_position_world"], dtype=np.float32)
            goal_w = np.asarray(sample["goal_position_world"], dtype=np.float32)
            expected = np.asarray(sample["computed_target_position"], dtype=np.float32)
            quat_wxyz = _estimate_quaternion_wxyz(goal_w, robot_w, expected[:3])

            config = SimpleNamespace(navigation={"fixed_target_position": goal_w.tolist()})

            raw_state = SimpleNamespace(
                imu=SimpleNamespace(quaternion=quat_wxyz),
                high_state=SimpleNamespace(position=robot_w),
            )
            got = target_position(raw_state, config)
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_log_distance_matches_geometry(self):
        for sample in LAB_SAMPLES:
            robot_w = np.asarray(sample["robot_position_world"], dtype=np.float64)
            goal_w = np.asarray(sample["goal_position_world"], dtype=np.float64)
            expected_log_distance = float(sample["computed_target_position"][3])
            distance = float(np.linalg.norm(goal_w - robot_w))
            got_log_distance = float(np.log(distance + 1.0))
            self.assertAlmostEqual(got_log_distance, expected_log_distance, places=5)


if __name__ == "__main__":
    unittest.main()