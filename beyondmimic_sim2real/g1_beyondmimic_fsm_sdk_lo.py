#!/usr/bin/env python3
"""State-machine wrapper for BeyondMimic 23DoF sim2real-in-sim.

This file intentionally leaves the validated direct sim2sim and SDK scripts
untouched.  It reuses their ONNX observation, DDS, joint mapping and safety
helpers, and follows RoboMimic_Deploy's Passive/FixedPose/Policy control flow.

Global keyboard commands (no Enter required):
  Space = stand/stop request, P = policy, H = request a safe policy hold,
  D = damping, Q/Esc = quit

If the global keyboard listener is disabled or unavailable, terminal commands
fall back to s/p/h/d/q followed by Enter.

Gamepad (unitree_mujoco use_joystick=1 → LowState.wireless_remote):
  Start / L2+Up = stand/stop request
  A / R2+A = policy
  B / R2+B = safe policy hold
  Y / L2+B = damping
  Select(Back) = quit

This program is deliberately restricted to --network lo.  It is a simulator
validation tool, not a real-robot executable.
"""
from __future__ import annotations

import argparse
import contextlib
import enum
import io as text_io
import os
import queue
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

DEPLOY_DIR = Path(__file__).resolve().parent
# Repo root: .../23dof_sim2real ; Project root may contain sibling RoboMimic_Deploy.
REPO_ROOT = DEPLOY_DIR.parent
PROJECT_DIR = REPO_ROOT.parent
sys.path.insert(0, str(DEPLOY_DIR))
# Prefer vendored common/ (remote_controller, keyboard); optional sibling RoboMimic.
sys.path.insert(0, str(PROJECT_DIR / "RoboMimic_Deploy"))

import deploy_mujocofor23 as d  # noqa: E402
import g1_beyondmimic_sdk_lo as sdk  # noqa: E402
from common.remote_controller import KeyMap, RemoteController  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402
from unitree_sdk2py.utils.thread import RecurrentThread  # noqa: E402

try:
    from common.keyboard import Keyboard  # noqa: E402
except Exception as exc:  # pynput may be absent or have no desktop/X11 access.
    Keyboard = None  # type: ignore[assignment,misc]
    KEYBOARD_IMPORT_ERROR: Exception | None = exc
else:
    KEYBOARD_IMPORT_ERROR = None


class ControlState(enum.Enum):
    STAND_UP = "STAND_UP"
    READY_STAND = "READY_STAND"
    POLICY = "POLICY"
    POLICY_HOLD = "POLICY_HOLD"
    DAMPING = "DAMPING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BeyondMimic G1 23DoF FSM sim2real-in-sim"
    )
    parser.add_argument("--network", default="lo")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--onnx", type=Path, default=DEPLOY_DIR / "g1_play/policy.onnx")
    parser.add_argument("--motion", type=Path, default=DEPLOY_DIR / "g1_play/motion.npz")
    parser.add_argument("--stand_frame", type=int, default=10)
    parser.add_argument("--start_frame", type=int, default=10)
    parser.add_argument("--policy_kp_scale", type=float, default=1.0)
    parser.add_argument("--joint_limit_margin", type=float, default=0.03)
    parser.add_argument("--stand_s", type=float, default=3.0)
    parser.add_argument(
        "--policy_duration",
        type=float,
        default=0.0,
        help="seconds before requesting a safe stop; 0 runs until motion end",
    )
    parser.add_argument("--action_lowpass", type=float, default=0.8)
    parser.add_argument(
        "--no_global_keyboard",
        action="store_true",
        help="disable pynput hotkeys and use terminal commands followed by Enter",
    )
    return parser.parse_args()


def tick_delta_ms(now: int, then: int) -> int:
    return (int(now) - int(then)) & 0xFFFFFFFF


def start_console_reader(commands: queue.SimpleQueue[str]) -> None:
    def reader() -> None:
        while True:
            try:
                value = input().strip().lower()
            except EOFError:
                return
            if value:
                commands.put(value[0])
            if value.startswith("q"):
                return

    threading.Thread(target=reader, name="fsm-console", daemon=True).start()


def main() -> None:
    args = parse_args()
    if args.network != "lo":
        raise SystemExit(
            "This FSM file is sim2real-in-sim only; use --network lo. "
            "Real-robot deployment needs a separate safety review."
        )
    if args.joint_limit_margin < 0.0:
        raise SystemExit("--joint_limit_margin must be >= 0")
    if args.stand_s <= 0.0:
        raise SystemExit("--stand_s must be > 0")

    cyclonedds_home = REPO_ROOT / "cyclonedds" / "install"
    if not cyclonedds_home.is_dir():
        cyclonedds_home = PROJECT_DIR / "23dof_sim2real" / "cyclonedds" / "install"
    if cyclonedds_home.is_dir():
        os.environ["CYCLONEDDS_HOME"] = str(cyclonedds_home)

    joint_seq, default_seq, stiffness_seq, damping_seq, action_scale_seq = (
        d.load_onnx_metadata(str(args.onnx))
    )
    stiffness_xml = (
        d.policy_to_xml(stiffness_seq, joint_seq) * float(args.policy_kp_scale)
    )
    damping_xml = d.policy_to_xml(damping_seq, joint_seq) * (
        float(args.policy_kp_scale) ** 0.5
    )
    stand_kp = np.full(d.NUM_ACTIONS, 200.0, dtype=np.float32)
    stand_kd = np.full(d.NUM_ACTIONS, 5.0, dtype=np.float32)
    damping_kp = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    damping_kd = np.full(d.NUM_ACTIONS, 8.0, dtype=np.float32)

    motion = np.load(args.motion)
    max_timestep = int(motion["joint_pos"].shape[0])
    stand_frame = int(np.clip(args.stand_frame, 0, max_timestep - 1))
    start_frame = int(np.clip(args.start_frame, 0, max_timestep - 1))
    stand_raw = d.policy_to_xml(
        motion["joint_pos"][stand_frame].astype(np.float32), joint_seq
    )
    stand_q, stand_clip = sdk.clip_joint_targets(
        stand_raw, args.joint_limit_margin
    )
    if np.any(stand_clip):
        names = ", ".join(np.asarray(d.JOINT_XML)[stand_clip])
        print(f"[WARN] stand target clipped: {names}")

    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 1
    ort_options.inter_op_num_threads = 1
    ort_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(args.onnx), sess_options=ort_options, providers=["CPUExecutionProvider"]
    )
    warm_obs = np.zeros((1, d.NUM_OBS), dtype=np.float32)
    warm_step = np.zeros((1, 1), dtype=np.float32)
    for _ in range(10):
        session.run(["actions"], {"obs": warm_obs, "time_step": warm_step})

    ChannelFactoryInitialize(int(args.domain_id), args.network)
    io = sdk.RobotIO()
    print(
        f"[fsm] waiting for LowState + SportModeState on "
        f"{args.network} domain={args.domain_id} ..."
    )
    io.wait()
    low, sport = io.snapshot()
    assert low is not None and sport is not None
    q_xml, _ = sdk.xml_from_lowstate(low)

    command_lock = threading.Lock()
    q_cmd, kp_cmd, kd_cmd = sdk.pack_full(q_xml, stand_kp, stand_kd)

    def write_loop() -> None:
        with command_lock:
            q = q_cmd.copy()
            kp = kp_cmd.copy()
            kd = kd_cmd.copy()
        state, _ = io.snapshot()
        mode_machine = int(state.mode_machine) if state is not None else 4
        io.send(q, kp, kd, mode_machine)

    writer = RecurrentThread(
        interval=sdk.CMD_DT, target=write_loop, name="fsm-lowcmd"
    )
    writer.Start()

    console_commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    keyboard = None
    if not args.no_global_keyboard and Keyboard is not None:
        try:
            # RoboMimic's shared Keyboard prints its own unrelated skill map.
            # Hide that text and print the bindings used by this FSM below.
            with contextlib.redirect_stdout(text_io.StringIO()):
                keyboard = Keyboard()
        except Exception as exc:
            print(f"[WARN] global keyboard unavailable ({exc}); using terminal input")
    elif not args.no_global_keyboard and KEYBOARD_IMPORT_ERROR is not None:
        print(
            f"[WARN] global keyboard unavailable ({KEYBOARD_IMPORT_ERROR}); "
            "using terminal input"
        )
    if keyboard is None:
        start_console_reader(console_commands)

    remote = RemoteController()
    previous_combos: dict[str, bool] = {}

    state = ControlState.STAND_UP
    state_entry_tick = int(low.tick) & 0xFFFFFFFF
    transition_q0 = q_xml.copy()
    ready_q = stand_q.copy()
    last_control_tick: int | None = None
    last_seen_tick = state_entry_tick
    last_tick_wall = time.monotonic()
    running = True

    motion_pos: np.ndarray | None = None
    motion_quat: np.ndarray | None = None
    policy_start_tick = state_entry_tick
    timestep = start_frame
    policy_updates = 0
    skipped_frames = 0
    limit_clip_events = 0
    limit_clip_max = 0.0
    stop_requested = False
    action_buffer = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    action_filtered = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    obs = np.zeros(d.NUM_OBS, dtype=np.float32)
    alpha = float(np.clip(args.action_lowpass, 0.0, 1.0))

    print("[fsm] STAND_UP: moving to motion start pose")
    if keyboard is not None:
        print(
            "[fsm] global keyboard (no Enter): "
            "Space=stand/stop P=policy H=hold D=damping Q/Esc=quit"
        )
    else:
        print("[fsm] terminal: s=stand p=policy h=hold d=damping q=quit + Enter")
    print(
        "[fsm] gamepad: Start/L2+Up=stand  A/R2+A=policy  "
        "B/R2+B=hold  Y/L2+B=damping  Select=quit"
    )

    def publish_target(
        target: np.ndarray, kp: np.ndarray, kd: np.ndarray
    ) -> None:
        nonlocal q_cmd, kp_cmd, kd_cmd
        packed = sdk.pack_full(target, kp, kd)
        with command_lock:
            q_cmd[:], kp_cmd[:], kd_cmd[:] = packed

    def transition(
        new_state: ControlState,
        tick: int,
        current_q: np.ndarray,
        torso_pos: np.ndarray,
        torso_quat: np.ndarray,
        reason: str,
    ) -> None:
        nonlocal state, state_entry_tick, transition_q0, ready_q
        nonlocal motion_pos, motion_quat, policy_start_tick, timestep
        nonlocal policy_updates, skipped_frames, limit_clip_events, limit_clip_max
        nonlocal action_buffer, action_filtered, obs, stop_requested

        if new_state == state:
            return
        old_state = state
        state = new_state
        state_entry_tick = tick
        transition_q0 = current_q.copy()

        if new_state == ControlState.POLICY:
            motion_pos, motion_quat = sdk.align_motion_to_robot(
                motion["body_pos_w"],
                motion["body_quat_w"],
                motion["body_pos_w"][start_frame, d.TORSO_BODY_INDEX],
                motion["body_quat_w"][start_frame, d.TORSO_BODY_INDEX],
                torso_pos,
                torso_quat,
            )
            policy_start_tick = tick
            timestep = start_frame
            policy_updates = 0
            skipped_frames = 0
            limit_clip_events = 0
            limit_clip_max = 0.0
            stop_requested = False
            action_buffer[:] = 0.0
            action_filtered[:] = 0.0
            obs[:] = 0.0
        elif new_state == ControlState.READY_STAND and old_state == ControlState.STAND_UP:
            ready_q = stand_q.copy()
        elif old_state == ControlState.POLICY:
            print(
                f"[fsm] policy summary: updates={policy_updates} frame={timestep} "
                f"skipped={skipped_frames} limit_clips={limit_clip_events} "
                f"clip_max={limit_clip_max:.3f}rad"
            )

        print(f"[fsm] {old_state.value} -> {new_state.value}: {reason}")

    try:
        while running:
            low, sport = io.snapshot()
            if low is None or sport is None:
                time.sleep(0.001)
                continue

            tick = int(low.tick) & 0xFFFFFFFF
            if tick != last_seen_tick:
                last_seen_tick = tick
                last_tick_wall = time.monotonic()
            elif time.monotonic() - last_tick_wall > 0.5:
                print("[SAFETY] LowState robot clock stale for >0.5s")
                q_now, _ = sdk.xml_from_lowstate(low)
                publish_target(q_now, damping_kp, damping_kd)
                break

            if (
                last_control_tick is not None
                and tick_delta_ms(tick, last_control_tick) < sdk.POLICY_TICK_MS
            ):
                time.sleep(0.001)
                continue
            last_control_tick = tick

            q_xml, dq_xml = sdk.xml_from_lowstate(low)
            imu_quat = np.asarray(low.imu_state.quaternion, dtype=np.float64)
            imu_quat /= max(np.linalg.norm(imu_quat), 1e-8)
            gyro = np.asarray(low.imu_state.gyroscope, dtype=np.float64)
            pelvis_pos, pelvis_quat, pelvis_lin_w = sdk.imu_to_pelvis(
                np.asarray(sport.position, dtype=np.float64),
                imu_quat,
                np.asarray(sport.velocity, dtype=np.float64),
                gyro,
            )
            waist = float(q_xml[d.JOINT_XML.index("waist_yaw_joint")])
            torso_pos, torso_quat = sdk.pelvis_to_torso(
                pelvis_pos, pelvis_quat, waist
            )
            rpy = np.asarray(low.imu_state.rpy, dtype=np.float64)

            try:
                remote.set(bytes(low.wireless_remote))
            except (AttributeError, TypeError, ValueError):
                pass

            # Xbox-friendly singles (A/B/Y) plus RoboMimic-style combos.
            combos = {
                "stand": remote.is_button_pressed(KeyMap.start)
                or (
                    remote.is_button_pressed(KeyMap.L2)
                    and remote.is_button_pressed(KeyMap.up)
                ),
                "policy": remote.is_button_pressed(KeyMap.A)
                or (
                    remote.is_button_pressed(KeyMap.R2)
                    and remote.is_button_pressed(KeyMap.A)
                ),
                "hold": remote.is_button_pressed(KeyMap.B)
                or (
                    remote.is_button_pressed(KeyMap.R2)
                    and remote.is_button_pressed(KeyMap.B)
                ),
                "damping": remote.is_button_pressed(KeyMap.Y)
                or remote.is_button_pressed(KeyMap.F1)
                or (
                    remote.is_button_pressed(KeyMap.L2)
                    and remote.is_button_pressed(KeyMap.B)
                ),
                "quit": remote.is_button_pressed(KeyMap.select),
            }
            remote_events = {
                name
                for name, active in combos.items()
                if active and not previous_combos.get(name, False)
            }
            for name in sorted(remote_events):
                print(f"[fsm] gamepad event: {name}")
            previous_combos = combos

            console_event = ""
            while not console_commands.empty():
                console_event = console_commands.get()

            keyboard_event = ""
            if keyboard is not None:
                keyboard.update()
                if keyboard.is_key_pressed("ESCAPE") or keyboard.is_key_released("Q"):
                    keyboard_event = "q"
                elif keyboard.is_key_released("D"):
                    keyboard_event = "d"
                elif keyboard.is_key_released("SPACE"):
                    keyboard_event = "s"
                elif keyboard.is_key_released("H"):
                    keyboard_event = "h"
                elif keyboard.is_key_released("P"):
                    keyboard_event = "p"

            if console_event == "q" or keyboard_event == "q" or "quit" in remote_events:
                running = False
                continue
            if console_event == "d" or keyboard_event == "d" or "damping" in remote_events:
                transition(
                    ControlState.DAMPING,
                    tick,
                    q_xml,
                    torso_pos,
                    torso_quat,
                    "operator damping command",
                )
            elif console_event == "s" or keyboard_event == "s" or "stand" in remote_events:
                if state == ControlState.POLICY:
                    if not stop_requested:
                        print("[fsm] policy stop requested; waiting for a low-motion frame")
                    stop_requested = True
                elif state == ControlState.POLICY_HOLD:
                    print(
                        "[fsm] fixed-pose return is unavailable without a "
                        "separate balance policy; press p to restart or d for damping"
                    )
                elif state == ControlState.DAMPING and (
                    pelvis_pos[2] < 0.60
                    or abs(rpy[0]) > 0.35
                    or abs(rpy[1]) > 0.35
                ):
                    print(
                        "[fsm] stand ignored: robot is not upright; reset/support "
                        "the simulator before commanding stand"
                    )
                else:
                    transition(
                        ControlState.STAND_UP,
                        tick,
                        q_xml,
                        torso_pos,
                        torso_quat,
                        "operator stand command",
                    )
            elif console_event == "h" or keyboard_event == "h" or "hold" in remote_events:
                if state == ControlState.POLICY:
                    if not stop_requested:
                        print("[fsm] policy stop requested; waiting for a low-motion frame")
                    stop_requested = True
            elif console_event == "p" or keyboard_event == "p" or "policy" in remote_events:
                if state in (ControlState.READY_STAND, ControlState.POLICY_HOLD):
                    transition(
                        ControlState.POLICY,
                        tick,
                        q_xml,
                        torso_pos,
                        torso_quat,
                        "operator policy start",
                    )
                else:
                    print(f"[fsm] policy start ignored in {state.value}")

            fallen = (
                pelvis_pos[2] < 0.45
                or abs(rpy[0]) > 1.0
                or abs(rpy[1]) > 1.0
            )
            if fallen and state != ControlState.DAMPING:
                transition(
                    ControlState.DAMPING,
                    tick,
                    q_xml,
                    torso_pos,
                    torso_quat,
                    f"fall detected z={pelvis_pos[2]:.3f} "
                    f"roll={rpy[0]:.3f} pitch={rpy[1]:.3f}",
                )

            if state == ControlState.DAMPING:
                publish_target(q_xml, damping_kp, damping_kd)
                continue

            if state == ControlState.STAND_UP:
                elapsed = tick_delta_ms(tick, state_entry_tick) * 1e-3
                blend = float(np.clip(elapsed / args.stand_s, 0.0, 1.0))
                target = (1.0 - blend) * transition_q0 + blend * stand_q
                publish_target(target, stand_kp, stand_kd)
                if blend >= 1.0:
                    transition(
                        ControlState.READY_STAND,
                        tick,
                        q_xml,
                        torso_pos,
                        torso_quat,
                        "stand interpolation complete",
                    )
                continue

            if state == ControlState.READY_STAND:
                publish_target(ready_q, stand_kp, stand_kd)
                continue

            assert state in (ControlState.POLICY, ControlState.POLICY_HOLD)
            assert motion_pos is not None and motion_quat is not None
            if state == ControlState.POLICY:
                policy_elapsed_ms = tick_delta_ms(tick, policy_start_tick)
                if (
                    args.policy_duration > 0.0
                    and not stop_requested
                    and policy_elapsed_ms >= int(round(args.policy_duration * 1000.0))
                ):
                    stop_requested = True
                    print("[fsm] policy duration reached; waiting for a low-motion frame")

                scheduled_timestep = start_frame + policy_elapsed_ms // sdk.POLICY_TICK_MS
                if scheduled_timestep >= max_timestep:
                    timestep = max_timestep - 1
                    transition(
                        ControlState.POLICY_HOLD,
                        tick,
                        q_xml,
                        torso_pos,
                        torso_quat,
                        "motion complete; freezing final reference",
                    )
                    continue
                if scheduled_timestep - timestep > 5:
                    print(
                        f"[WARN] robot clock discontinuity: "
                        f"{scheduled_timestep - timestep} frames; entering damping"
                    )
                    transition(
                        ControlState.DAMPING,
                        tick,
                        q_xml,
                        torso_pos,
                        torso_quat,
                        "clock discontinuity",
                    )
                    continue
                if scheduled_timestep > timestep:
                    skipped_frames += max(0, scheduled_timestep - timestep)
                    timestep = int(scheduled_timestep)

            if state == ControlState.POLICY and stop_requested:
                ref_joint_speed = float(
                    np.sqrt(np.mean(motion["joint_vel"][timestep] ** 2))
                )
                ref_base_speed = float(
                    np.linalg.norm(motion["body_lin_vel_w"][timestep, 0])
                )
                ref_base_angular_speed = float(
                    np.linalg.norm(motion["body_ang_vel_w"][timestep, 0])
                )
                measured_joint_speed = float(np.sqrt(np.mean(dq_xml * dq_xml)))
                measured_tilt_rate = float(np.linalg.norm(gyro[:2]))
                safe_to_freeze = (
                    ref_joint_speed < 0.45
                    and ref_base_speed < 0.30
                    and ref_base_angular_speed < 0.35
                    and measured_joint_speed < 1.0
                    and measured_tilt_rate < 0.7
                    and pelvis_pos[2] > 0.60
                    and abs(rpy[0]) < 0.35
                    and abs(rpy[1]) < 0.35
                )
                if safe_to_freeze:
                    transition(
                        ControlState.POLICY_HOLD,
                        tick,
                        q_xml,
                        torso_pos,
                        torso_quat,
                        "low-motion frame reached; reference frozen",
                    )
                    continue

            motion_cmd = np.concatenate(
                [motion["joint_pos"][timestep], motion["joint_vel"][timestep]]
            ).astype(np.float32)
            anchor_pos, anchor_quat = d.subtract_frame_transforms_mujoco(
                torso_pos.astype(np.float32),
                torso_quat.astype(np.float32),
                motion_pos[timestep, d.TORSO_BODY_INDEX],
                motion_quat[timestep, d.TORSO_BODY_INDEX],
            )
            anchor_matrix = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(anchor_matrix, anchor_quat)
            anchor_ori = (
                anchor_matrix.reshape(3, 3)[:, :2].reshape(-1).astype(np.float32)
            )
            base_lin = d.quat_rotate_inverse_np(
                pelvis_quat, pelvis_lin_w
            ).astype(np.float32)
            q_policy = d.xml_to_policy(q_xml, joint_seq)
            dq_policy = d.xml_to_policy(dq_xml, joint_seq)

            offset = 0
            obs[offset : offset + 46] = motion_cmd
            offset += 46
            obs[offset : offset + 3] = anchor_pos
            offset += 3
            obs[offset : offset + 6] = anchor_ori
            offset += 6
            obs[offset : offset + 3] = base_lin
            offset += 3
            obs[offset : offset + 3] = gyro.astype(np.float32)
            offset += 3
            obs[offset : offset + d.NUM_ACTIONS] = q_policy - default_seq
            offset += d.NUM_ACTIONS
            obs[offset : offset + d.NUM_ACTIONS] = dq_policy
            offset += d.NUM_ACTIONS
            obs[offset : offset + d.NUM_ACTIONS] = action_buffer

            action = session.run(
                ["actions"],
                {
                    "obs": obs[None, :].astype(np.float32),
                    "time_step": np.array([[float(timestep)]], dtype=np.float32),
                },
            )[0]
            action = np.clip(
                np.asarray(action, dtype=np.float32).reshape(-1), -5.0, 5.0
            )
            action_buffer = action.copy()
            action_filtered = alpha * action + (1.0 - alpha) * action_filtered
            raw_target = d.policy_to_xml(
                action_filtered * action_scale_seq + default_seq, joint_seq
            )
            target, clip_mask = sdk.clip_joint_targets(
                raw_target, args.joint_limit_margin
            )
            limit_clip_events += int(np.count_nonzero(clip_mask))
            limit_clip_max = max(
                limit_clip_max, float(np.max(np.abs(raw_target - target)))
            )
            publish_target(target, stiffness_xml, damping_xml)
            policy_updates += 1
            if state == ControlState.POLICY:
                timestep += 1

            if policy_updates % 50 == 0:
                q_error = target - q_xml
                print(
                    f"[fsm] {state.value} frame={timestep} z={pelvis_pos[2]:.3f} "
                    f"|anchor|={np.linalg.norm(anchor_pos):.3f} "
                    f"qerr_rms={np.sqrt(np.mean(q_error * q_error)):.3f} "
                    f"limit_clips={limit_clip_events} "
                    f"clip_max={limit_clip_max:.3f}rad"
                )
    except KeyboardInterrupt:
        print("[fsm] interrupted")
    finally:
        if keyboard is not None:
            keyboard.stop()
        low, _ = io.snapshot()
        if low is not None:
            q_xml, _ = sdk.xml_from_lowstate(low)
            publish_target(q_xml, damping_kp, damping_kd)
        time.sleep(0.2)
        writer.Wait(timeout=1.0)
        print("[fsm] stopped in damping mode")


if __name__ == "__main__":
    main()
