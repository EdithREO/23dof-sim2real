#!/usr/bin/env python3
"""Safe power-on only (no policy).

State flow:
  ZERO_TORQUE --Start--> STAND_UP --done--> READY_STAND
       |                     |                  |
       +---------- Y / Select / Ctrl+C -------->+--> DAMPING -> exit

This does NOT run the ONNX policy. Pressing A only prints a reminder.
Use after loco is off and the robot is in debug mode (real) or with
unitree_mujoco on lo (sim).

Keyboard (no Enter if global listener works):
  Space = Start (leave zero torque)
  D / Y-equivalent = damping
  Q / Esc = quit via damping

Gamepad (LowState.wireless_remote):
  Start = leave zero torque
  Y or Select = damping / quit
  A = ignored (policy not enabled yet)
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

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent
PROJECT_DIR = REPO_ROOT.parent
sys.path.insert(0, str(DEPLOY_DIR))
sys.path.insert(0, str(PROJECT_DIR / "RoboMimic_Deploy"))

import deploy_mujocofor23 as d  # noqa: E402
import g1_beyondmimic_sdk_lo as sdk  # noqa: E402
from common.remote_controller import KeyMap, RemoteController  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402
from unitree_sdk2py.utils.thread import RecurrentThread  # noqa: E402

try:
    from common.keyboard import Keyboard  # noqa: E402
except Exception as exc:
    Keyboard = None  # type: ignore[assignment,misc]
    KEYBOARD_IMPORT_ERROR: Exception | None = exc
else:
    KEYBOARD_IMPORT_ERROR = None


class PowerState(enum.Enum):
    ZERO_TORQUE = "ZERO_TORQUE"
    STAND_UP = "STAND_UP"
    READY_STAND = "READY_STAND"
    DAMPING = "DAMPING"


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

    threading.Thread(target=reader, name="power-console", daemon=True).start()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 23DoF safe power-on (no policy)")
    p.add_argument("--network", default="lo", help="DDS iface; real robot needs --allow-real")
    p.add_argument("--domain_id", type=int, default=0)
    p.add_argument("--allow-real", action="store_true", help="required when network is not lo")
    p.add_argument("--onnx", type=Path, default=DEPLOY_DIR / "g1_play/policy.onnx")
    p.add_argument("--motion", type=Path, default=DEPLOY_DIR / "g1_play/motion.npz")
    p.add_argument("--stand_frame", type=int, default=10)
    p.add_argument("--stand_s", type=float, default=2.0)
    p.add_argument("--joint_limit_margin", type=float, default=0.03)
    p.add_argument(
        "--no_global_keyboard",
        action="store_true",
        help="use terminal s/d/q + Enter instead of global hotkeys",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.network != "lo" and not args.allow_real:
        raise SystemExit("非 lo 网卡必须加 --allow-real（真机吊带+急停就绪后再用）。")
    if args.stand_s <= 0.0:
        raise SystemExit("--stand_s must be > 0")
    if args.joint_limit_margin < 0.0:
        raise SystemExit("--joint_limit_margin must be >= 0")
    if args.network != "lo" and args.allow_real:
        print("[WARN] REAL ROBOT power-on. Hang + e-stop ready. Starting in 3s...")
        time.sleep(3.0)

    cyclonedds_home = REPO_ROOT / "cyclonedds" / "install"
    if not cyclonedds_home.is_dir():
        cyclonedds_home = PROJECT_DIR / "23dof_sim2real" / "cyclonedds" / "install"
    if cyclonedds_home.is_dir():
        os.environ["CYCLONEDDS_HOME"] = str(cyclonedds_home)

    joint_seq, _default, _stiff, _damp, _ascale = d.load_onnx_metadata(str(args.onnx))
    motion = np.load(args.motion)
    stand_frame = int(np.clip(args.stand_frame, 0, motion["joint_pos"].shape[0] - 1))
    stand_raw = d.policy_to_xml(
        motion["joint_pos"][stand_frame].astype(np.float32), joint_seq
    )
    stand_q, stand_clip = sdk.clip_joint_targets(stand_raw, args.joint_limit_margin)
    if np.any(stand_clip):
        names = ", ".join(np.asarray(d.JOINT_XML)[stand_clip])
        print(f"[WARN] stand target clipped: {names}")

    zero_kp = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    zero_kd = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    stand_kp = np.full(d.NUM_ACTIONS, 200.0, dtype=np.float32)
    stand_kd = np.full(d.NUM_ACTIONS, 5.0, dtype=np.float32)
    damping_kp = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    damping_kd = np.full(d.NUM_ACTIONS, 8.0, dtype=np.float32)

    ChannelFactoryInitialize(int(args.domain_id), args.network)
    io = sdk.RobotIO()
    print(
        f"[power] waiting LowState on {args.network} domain={args.domain_id} "
        "(sportmodestate not required) ..."
    )
    io.wait(need_sport=False)
    low, _ = io.snapshot()
    assert low is not None
    q_xml, _ = sdk.xml_from_lowstate(low)

    command_lock = threading.Lock()
    q_cmd, kp_cmd, kd_cmd = sdk.pack_full(q_xml, zero_kp, zero_kd)

    def write_loop() -> None:
        with command_lock:
            q = q_cmd.copy()
            kp = kp_cmd.copy()
            kd = kd_cmd.copy()
        state, _ = io.snapshot()
        mode_machine = int(state.mode_machine) if state is not None else 4
        io.send(q, kp, kd, mode_machine)

    writer = RecurrentThread(interval=sdk.CMD_DT, target=write_loop, name="power-lowcmd")
    writer.Start()

    console_commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    keyboard = None
    if not args.no_global_keyboard and Keyboard is not None:
        try:
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

    state = PowerState.ZERO_TORQUE
    state_entry_tick = int(low.tick) & 0xFFFFFFFF
    transition_q0 = q_xml.copy()
    last_control_tick: int | None = None
    last_seen_tick = state_entry_tick
    last_tick_wall = time.monotonic()
    last_status_wall = 0.0
    running = True
    exit_after_damping = False

    def publish_target(target: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        nonlocal q_cmd, kp_cmd, kd_cmd
        packed = sdk.pack_full(target, kp, kd)
        with command_lock:
            q_cmd[:], kp_cmd[:], kd_cmd[:] = packed

    def transition(new_state: PowerState, tick: int, current_q: np.ndarray, reason: str) -> None:
        nonlocal state, state_entry_tick, transition_q0
        if new_state == state:
            return
        old = state
        state = new_state
        state_entry_tick = tick
        transition_q0 = current_q.copy()
        print(f"[power] {old.value} -> {new_state.value}: {reason}")

    print("[power] ZERO_TORQUE: motors unloaded; hold the robot")
    print("[power] Start/Space = stand up | Y/D = damping | Select/Q = quit")
    print("[power] A = policy NOT enabled in this script")

    try:
        while running:
            low, _ = io.snapshot()
            if low is None:
                time.sleep(0.001)
                continue

            tick = int(low.tick) & 0xFFFFFFFF
            if tick != last_seen_tick:
                last_seen_tick = tick
                last_tick_wall = time.monotonic()
            elif time.monotonic() - last_tick_wall > 0.5:
                print("[SAFETY] LowState robot clock stale for >0.5s -> damping")
                q_now, _ = sdk.xml_from_lowstate(low)
                publish_target(q_now, damping_kp, damping_kd)
                transition(PowerState.DAMPING, tick, q_now, "stale lowstate")
                # No fresh robot clock remains to advance the control loop.
                # Publish damping briefly, then leave instead of repeating this
                # branch forever with the same stale tick.
                time.sleep(0.2)
                running = False
                continue

            if (
                last_control_tick is not None
                and tick_delta_ms(tick, last_control_tick) < sdk.POLICY_TICK_MS
            ):
                time.sleep(0.001)
                continue
            last_control_tick = tick

            q_xml, _ = sdk.xml_from_lowstate(low)
            rpy = np.asarray(low.imu_state.rpy, dtype=np.float64)
            quat = np.asarray(low.imu_state.quaternion, dtype=np.float64)
            quat = quat / max(np.linalg.norm(quat), 1e-8)

            try:
                remote.set(bytes(low.wireless_remote))
            except (AttributeError, TypeError, ValueError):
                pass

            combos = {
                "start": remote.is_button_pressed(KeyMap.start),
                "damping": remote.is_button_pressed(KeyMap.Y)
                or remote.is_button_pressed(KeyMap.F1)
                or (
                    remote.is_button_pressed(KeyMap.L2)
                    and remote.is_button_pressed(KeyMap.B)
                ),
                "quit": remote.is_button_pressed(KeyMap.select),
                "policy_hint": remote.is_button_pressed(KeyMap.A),
            }
            remote_events = {
                name
                for name, active in combos.items()
                if active and not previous_combos.get(name, False)
            }
            for name in sorted(remote_events):
                print(f"[power] gamepad event: {name}")
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
                elif keyboard.is_key_released("A") or keyboard.is_key_released("P"):
                    keyboard_event = "a"

            want_start = (
                console_event == "s"
                or keyboard_event == "s"
                or "start" in remote_events
            )
            want_damp = (
                console_event == "d"
                or keyboard_event == "d"
                or "damping" in remote_events
            )
            want_quit = (
                console_event == "q"
                or keyboard_event == "q"
                or "quit" in remote_events
            )
            want_policy_hint = (
                console_event == "a"
                or keyboard_event == "a"
                or "policy_hint" in remote_events
            )

            if want_policy_hint:
                print("[power] policy not enabled yet; finish power-on first")

            if want_damp or want_quit:
                transition(
                    PowerState.DAMPING,
                    tick,
                    q_xml,
                    "operator damping/quit",
                )
                if want_quit:
                    exit_after_damping = True
                # Stay in the loop so DAMPING commands are actually published.

            if state == PowerState.ZERO_TORQUE:
                publish_target(q_xml, zero_kp, zero_kd)
                if want_start:
                    transition(
                        PowerState.STAND_UP,
                        tick,
                        q_xml,
                        "operator start -> slow stand",
                    )
                elif time.monotonic() - last_status_wall > 1.0:
                    last_status_wall = time.monotonic()
                    print(
                        f"[power] ZERO_TORQUE waiting Start | "
                        f"quat={quat} roll={rpy[0]:.3f} pitch={rpy[1]:.3f}"
                    )
                continue

            if state == PowerState.STAND_UP:
                elapsed = tick_delta_ms(tick, state_entry_tick) * 1e-3
                blend = float(np.clip(elapsed / args.stand_s, 0.0, 1.0))
                target = (1.0 - blend) * transition_q0 + blend * stand_q
                publish_target(target.astype(np.float32), stand_kp, stand_kd)
                if blend >= 1.0:
                    transition(
                        PowerState.READY_STAND,
                        tick,
                        q_xml,
                        "stand interpolation complete",
                    )
                continue

            if state == PowerState.READY_STAND:
                publish_target(stand_q, stand_kp, stand_kd)
                if time.monotonic() - last_status_wall > 1.0:
                    last_status_wall = time.monotonic()
                    upright = abs(rpy[0]) < 0.35 and abs(rpy[1]) < 0.35
                    print(
                        f"[power] READY_STAND quat={quat} "
                        f"roll={rpy[0]:.3f} pitch={rpy[1]:.3f} "
                        f"{'OK-ish upright' if upright else 'NOT upright — do not run policy'}"
                    )
                continue

            # DAMPING
            publish_target(q_xml, damping_kp, damping_kd)
            if exit_after_damping:
                time.sleep(0.2)
                running = False

    except KeyboardInterrupt:
        print("[power] interrupted -> damping")
        low, _ = io.snapshot()
        if low is not None:
            q_xml, _ = sdk.xml_from_lowstate(low)
            publish_target(q_xml, damping_kp, damping_kd)
            time.sleep(0.2)
    finally:
        if keyboard is not None:
            keyboard.stop()
        low, _ = io.snapshot()
        if low is not None:
            q_xml, _ = sdk.xml_from_lowstate(low)
            publish_target(q_xml, damping_kp, damping_kd)
        time.sleep(0.2)
        writer.Wait(timeout=1.0)
        print("[power] stopped in damping")


if __name__ == "__main__":
    main()
