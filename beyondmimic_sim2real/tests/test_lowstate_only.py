"""Offline regression tests; never initialize DDS or send robot commands."""
import ast
import contextlib
import io as text_io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
import g1_beyondmimic_fsm_sdk_lo as fsm
import g1_beyondmimic_sdk_lo as sdk


class LowStateOnlyTests(unittest.TestCase):
    def test_orientation_matches_previous_full_pose_algorithm(self):
        with np.load(DEPLOY / "g1_play/motion.npz") as motion:
            quats = motion["body_quat_w"]
            positions = motion["body_pos_w"]
        torso = fsm.d.TORSO_BODY_INDEX
        for yaw in (-2.0, 0.0, 1.3):
            current = fsm.d.quaternion_multiply(sdk.yaw_quat(yaw), quats[10, torso])
            for destination in (np.zeros(3), np.array([20., -10., 2.])):
                old_pos, old_quats = sdk.align_motion_to_robot(
                    positions, quats, positions[10, torso], quats[10, torso],
                    destination, current,
                )
                new_quats = fsm.align_reference_orientation(quats[:, torso], 10, current)
                np.testing.assert_allclose(new_quats, old_quats[:, torso], atol=1e-7)
                for frame in (0, 10, 484, 5000, len(quats) - 1):
                    _, old_rel = fsm.d.subtract_frame_transforms_mujoco(
                        destination, current, old_pos[frame, torso], old_quats[frame, torso]
                    )
                    new_rel = fsm.d.quaternion_multiply(
                        fsm.d.quaternion_conjugate(current), new_quats[frame]
                    )
                    new_rel /= np.linalg.norm(new_rel)
                    np.testing.assert_allclose(
                        sdk.quat_to_mat(new_rel)[:, :2].reshape(-1),
                        sdk.quat_to_mat(old_rel)[:, :2].reshape(-1), atol=2e-7,
                    )

    def test_low_only_subscribes_and_waits_without_sport(self):
        with patch.object(sdk, "ChannelPublisher"), patch.object(sdk, "ChannelSubscriber") as subscriber:
            io = sdk.RobotIO(subscribe_sport=False)
            self.assertEqual(subscriber.call_count, 1)
            self.assertEqual(subscriber.call_args.args[0], "rt/lowstate")
            self.assertIsNone(io.sport_sub)
            sample = object()
            io._on_low(sample)
            io.wait(timeout_s=0.1, need_sport=False)
            self.assertEqual(io.snapshot(), (sample, None))

    def test_legacy_robotio_subscription_is_unchanged(self):
        with patch.object(sdk, "ChannelPublisher"), patch.object(sdk, "ChannelSubscriber") as subscriber:
            sdk.RobotIO()
            self.assertEqual(subscriber.call_count, 2)
            self.assertEqual(subscriber.call_args.args[0], "rt/sportmodestate")

    def test_fsm_has_no_sport_or_height_dependency(self):
        tree = ast.parse(Path(fsm.__file__).read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertTrue(names.isdisjoint({"sport", "pelvis_pos", "pelvis_lin_w"}))

    def test_main_runs_policy_and_damping_with_no_sport(self):
        # Real ONNX inference, but fake clock/feedback/publisher and no threads.
        queue = None
        clock = 0

        def console_reader(commands):
            nonlocal queue
            queue = commands

        def snapshot():
            nonlocal clock
            clock += 20
            if queue is not None:
                if clock == 120:
                    queue.put("p")
                elif clock == 240:
                    queue.put("d")
                elif clock >= 280:
                    queue.put("q")
            low = SimpleNamespace(
                tick=clock, mode_machine=4,
                motor_state=[SimpleNamespace(q=0., dq=0.) for _ in range(29)],
                imu_state=SimpleNamespace(
                    quaternion=[1., 0., 0., 0.], gyroscope=[0., 0., 0.], rpy=[0., 0., 0.]
                ),
                wireless_remote=bytes(40),
            )
            return low, None

        output = text_io.StringIO()
        with patch.object(sys, "argv", ["fsm", "--no_global_keyboard", "--stand_s", "0.02",
                                       "--policy_blend_s", "0.02"]), patch.object(
            fsm, "ChannelFactoryInitialize"
        ), patch.object(fsm.sdk, "RobotIO") as robot_io, patch.object(
            fsm, "RecurrentThread"
        ), patch.object(fsm, "start_console_reader", console_reader), patch.object(
            fsm.time, "sleep"
        ), contextlib.redirect_stdout(output):
            robot_io.return_value.snapshot.side_effect = snapshot
            fsm.main()
            robot_io.assert_called_once_with(subscribe_sport=False)
            robot_io.return_value.wait.assert_called_once_with(need_sport=False)
            robot_io.return_value.send.assert_not_called()
        self.assertIn("READY_STAND -> POLICY_BLEND", output.getvalue())
        self.assertIn("POLICY_BLEND -> POLICY:", output.getvalue())
        self.assertIn("POLICY -> DAMPING", output.getvalue())


if __name__ == "__main__":
    unittest.main()
