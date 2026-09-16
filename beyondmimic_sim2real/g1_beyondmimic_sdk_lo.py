#!/usr/bin/env python3
"""BeyondMimic walk1 on unitree_mujoco via SDK/DDS (sim2real-in-sim).

This is not the RMG 327-dim script. It uses Tracking-Flat-G123-v0 (130-dim obs,
23 actions, time_step) and talks LowState/LowCmd on --network lo.

  Terminal 1:  ./unitree_mujoco   (already configured for g1 / lo)
  Terminal 2:  python g1_beyondmimic_sdk_lo.py --network lo
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent
SDK_PY = REPO_ROOT / "unitree_sdk2_python"
sys.path.insert(0, str(DEPLOY_DIR))
if SDK_PY.is_dir():
    sys.path.insert(0, str(SDK_PY))

from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402
from unitree_sdk2py.utils.thread import RecurrentThread  # noqa: E402

import deploy_mujocofor23 as d  # noqa: E402

G1_NUM_MOTOR = 29
# 23DoF G1 in unitree_hg motor slots (skip waist roll/pitch and wrist pitch/yaw).
XML_TO_MOTOR = (
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
)
# unitree_mujoco g1_23dof.xml: IMU site on pelvis, torso_link child of pelvis.
IMU_IN_PELVIS = np.array([0.04525, 0.0, -0.08339], dtype=np.float64)
TORSO_IN_PELVIS = np.array([-0.0039635, 0.0, 0.054], dtype=np.float64)
POLICY_DT = 0.02
POLICY_TICK_MS = int(round(POLICY_DT * 1000.0))
# The Python DDS publisher and two subscribers share the GIL. Publishing the
# same target at 500 Hz starves LowState callbacks for seconds on loopback.
# 200 Hz keeps the low-level watchdog alive while leaving DDS headroom for state.
CMD_DT = 0.005
# The three-second preparation already settles the reference pose. An extra
# frozen-reference hold lets the free root drift before motion tracking starts.
HOLD_POLICY_STEPS = 0
XML_MOTOR_SET = set(XML_TO_MOTOR)

# Position limits in deploy_mujocofor23.JOINT_XML order.  These are the limits
# from unitree_mujoco/unitree_robots/g1/g1_23dof.xml (radians), which mirrors
# the G1 23DoF low-level joint layout used by this deployment script.  Limit
# the position target here rather than changing the Unitree SDK: clipping the
# network output to [-5, 5] does not guarantee that
# default_joint_pos + action * action_scale is mechanically reachable.
JOINT_POS_LOWER = np.array(
    [
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.618,
        -3.0892, -1.5882, -2.618, -1.0472, -1.97222,
        -3.0892, -2.2515, -2.618, -1.0472, -1.97222,
    ],
    dtype=np.float32,
)
JOINT_POS_UPPER = np.array(
    [
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
        2.618,
        2.6704, 2.2515, 2.618, 2.0944, 1.97222,
        2.6704, 1.5882, 2.618, 2.0944, 1.97222,
    ],
    dtype=np.float32,
)


def clip_joint_targets(q_des: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Clamp XML-order position targets inside the G1 joint limits."""
    lower = JOINT_POS_LOWER + margin
    upper = JOINT_POS_UPPER - margin
    if np.any(lower >= upper):
        raise ValueError(f"joint limit margin {margin} rad is too large")
    clipped = np.clip(np.asarray(q_des, dtype=np.float32), lower, upper)
    return clipped, np.abs(clipped - q_des) > 1e-6


def yaw_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(q, dtype=np.float64))
    return mat.reshape(3, 3)


def imu_to_pelvis(imu_pos: np.ndarray, imu_quat: np.ndarray, imu_vel_w: np.ndarray, gyro: np.ndarray):
    """Convert pelvis-mounted IMU site pose/vel to pelvis origin (site has no extra rotation)."""
    rot = quat_to_mat(imu_quat)
    offset = rot @ IMU_IN_PELVIS
    pelvis_pos = np.asarray(imu_pos, dtype=np.float64) - offset
    omega_w = rot @ np.asarray(gyro, dtype=np.float64)
    pelvis_vel_w = np.asarray(imu_vel_w, dtype=np.float64) - np.cross(omega_w, offset)
    return pelvis_pos, np.asarray(imu_quat, dtype=np.float64), pelvis_vel_w


def pelvis_to_torso(pelvis_pos: np.ndarray, pelvis_quat: np.ndarray, waist_yaw: float):
    rot = quat_to_mat(pelvis_quat)
    torso_pos = np.asarray(pelvis_pos, dtype=np.float64) + rot @ TORSO_IN_PELVIS
    torso_quat = d.quaternion_multiply(pelvis_quat, yaw_quat(waist_yaw))
    torso_quat = torso_quat / np.linalg.norm(torso_quat)
    return torso_pos, torso_quat


def align_motion_to_robot(motion_pos, motion_quat, src_pos, src_quat, dst_pos, dst_quat):
    """Yaw+XY align a clip so src frame matches the robot pose (keep z)."""
    dyaw = yaw_wxyz(dst_quat) - yaw_wxyz(src_quat)
    r = rot_z(dyaw)
    q_delta = yaw_quat(dyaw)
    src_xy = np.array(src_pos[:2], dtype=np.float64)
    dst_xy = np.array(dst_pos[:2], dtype=np.float64)
    rel = motion_pos.astype(np.float64).copy()
    rel[..., :2] -= src_xy
    flat = rel.reshape(-1, 3)
    flat[:] = (r @ flat.T).T
    rel = flat.reshape(motion_pos.shape)
    rel[..., :2] += dst_xy
    # Vectorized q_delta * motion_quat. The clip contains hundreds of thousands
    # of body quaternions; a Python nested loop stalls the control thread for
    # several seconds at handoff.
    q = motion_quat.astype(np.float64)
    dw, dx, dy, dz = q_delta
    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    out_q = np.stack(
        (
            dw * qw - dx * qx - dy * qy - dz * qz,
            dw * qx + dx * qw + dy * qz - dz * qy,
            dw * qy - dx * qz + dy * qw + dz * qx,
            dw * qz + dx * qy - dy * qx + dz * qw,
        ),
        axis=-1,
    )
    out_q /= np.maximum(np.linalg.norm(out_q, axis=-1, keepdims=True), 1e-12)
    return rel.astype(np.float32), out_q.astype(np.float32)


class RobotIO:
    def __init__(self):
        self._lock = threading.Lock()
        self.low_state: LowStateHG | None = None
        self.sport: SportModeState_ | None = None
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.pub.Init()
        self.low_sub = ChannelSubscriber("rt/lowstate", LowStateHG)
        # Do not add the SDK's BQueue here. When it fills, BQueue drops new
        # samples and preserves stale ones, which is unsuitable for feedback.
        # These callbacks only replace a reference under a lock, so running
        # them directly in the DDS listener is both safe and low latency.
        self.low_sub.Init(self._on_low)
        self.sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sport_sub.Init(self._on_sport)

    def _on_low(self, msg: LowStateHG):
        with self._lock:
            self.low_state = msg

    def _on_sport(self, msg: SportModeState_):
        with self._lock:
            self.sport = msg

    def wait(self, timeout_s: float = 20.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self.low_state is not None and self.sport is not None:
                    return
            time.sleep(0.01)
        with self._lock:
            got_low = self.low_state is not None
            got_sport = self.sport is not None
        raise TimeoutError(
            f"DDS timeout: lowstate={got_low} sportmodestate={got_sport}. "
            "先开 unitree_mujoco，确认 interface=lo 且 domain_id 一致。"
        )

    def snapshot(self):
        with self._lock:
            return self.low_state, self.sport

    def send(self, q_full, kp_full, kd_full, mode_machine: int) -> None:
        cmd = self.low_cmd
        cmd.mode_pr = 0
        cmd.mode_machine = int(mode_machine)
        for i in range(G1_NUM_MOTOR):
            m = cmd.motor_cmd[i]
            if i in XML_MOTOR_SET:
                m.mode = 1
                m.q = float(q_full[i])
                m.dq = 0.0
                m.kp = float(kp_full[i])
                m.kd = float(kd_full[i])
                m.tau = 0.0
            else:
                m.mode = 0
                m.q = 0.0
                m.dq = 0.0
                m.kp = 0.0
                m.kd = 0.0
                m.tau = 0.0
        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)


def xml_from_lowstate(msg: LowStateHG) -> tuple[np.ndarray, np.ndarray]:
    q = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    dq = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    for i, mid in enumerate(XML_TO_MOTOR):
        q[i] = float(msg.motor_state[mid].q)
        dq[i] = float(msg.motor_state[mid].dq)
    return q, dq


def tau_from_lowstate(msg: LowStateHG) -> np.ndarray:
    return np.asarray(
        [float(msg.motor_state[mid].tau_est) for mid in XML_TO_MOTOR],
        dtype=np.float32,
    )


def pack_full(q_xml: np.ndarray, kp_xml: np.ndarray, kd_xml: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_full = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    kp_full = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    kd_full = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    for i, mid in enumerate(XML_TO_MOTOR):
        q_full[mid] = q_xml[i]
        kp_full[mid] = kp_xml[i]
        kd_full[mid] = kd_xml[i]
    return q_full, kp_full, kd_full


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BeyondMimic G123 SDK sim2real-in-sim")
    p.add_argument("--network", default="lo")
    p.add_argument("--domain_id", type=int, default=0, help="DDS domain; 与 unitree_mujoco -i 一致")
    p.add_argument("--onnx", type=Path, default=DEPLOY_DIR / "g1_play/policy.onnx")
    p.add_argument("--motion", type=Path, default=DEPLOY_DIR / "g1_play/motion.npz")
    p.add_argument("--start_frame", type=int, default=10)
    p.add_argument(
        "--policy_kp_scale",
        type=float,
        default=1.0,
        help="policy 阶段相对 ONNX 训练增益的倍率（默认保持训练值 1.0）",
    )
    p.add_argument(
        "--joint_limit_margin",
        type=float,
        default=0.03,
        help="q_des 与机械限位之间保留的弧度余量（默认 0.03 rad）",
    )
    p.add_argument("--prepare_s", type=float, default=3.0)
    p.add_argument("--duration", type=float, default=25.0)
    p.add_argument("--allow-real", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.network != "lo" and not args.allow_real:
        raise SystemExit("非 lo 需要 --allow-real。仿真请用 --network lo。")
    if args.joint_limit_margin < 0.0:
        raise SystemExit("--joint_limit_margin 必须大于或等于 0。")

    cyclonedds_home = REPO_ROOT / "cyclonedds" / "install"
    if cyclonedds_home.is_dir():
        os.environ["CYCLONEDDS_HOME"] = str(cyclonedds_home)

    joint_seq, default_seq, stiffness_seq, damping_seq, action_scale_seq = d.load_onnx_metadata(str(args.onnx))
    stiffness_xml = d.policy_to_xml(stiffness_seq, joint_seq) * float(args.policy_kp_scale)
    damping_xml = d.policy_to_xml(damping_seq, joint_seq) * (float(args.policy_kp_scale) ** 0.5)
    motion = np.load(args.motion)
    start_frame = int(np.clip(args.start_frame, 0, motion["joint_pos"].shape[0] - 1))
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 1
    ort_options.inter_op_num_threads = 1
    ort_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(args.onnx), sess_options=ort_options, providers=["CPUExecutionProvider"]
    )
    # ONNX Runtime initializes some kernels lazily on the first run. Doing that
    # after policy handoff can hold one joint target for multiple robot seconds
    # and force a large motion-frame skip on the next update.
    warmup_obs = np.zeros((1, d.NUM_OBS), dtype=np.float32)
    warmup_step = np.zeros((1, 1), dtype=np.float32)
    warmup_start = time.monotonic()
    for _ in range(10):
        session.run(["actions"], {"obs": warmup_obs, "time_step": warmup_step})
    print(f"[INFO] ONNX warmup completed in {time.monotonic() - warmup_start:.3f}s")

    ChannelFactoryInitialize(int(args.domain_id), args.network)
    io = RobotIO()
    print(f"[sdk] waiting lowstate+sportmodestate on {args.network} domain={args.domain_id} ...")
    io.wait()
    low0, sport0 = io.snapshot()
    print(f"[sdk] connected mode_machine={low0.mode_machine}")
    imu_q = np.array(low0.imu_state.quaternion, dtype=np.float64)
    imu_q /= max(np.linalg.norm(imu_q), 1e-8)
    rpy = np.array(low0.imu_state.rpy, dtype=np.float64)
    q0, dq0 = xml_from_lowstate(low0)
    print(
        f"[sdk] sport.pos={np.array(sport0.position)} sport.vel={np.array(sport0.velocity)}\n"
        f"      imu.rpy={rpy} imu.quat={imu_q} q0[:6]={q0[:6]}"
    )

    # unitree_mujoco hold uses kp=200; taking over with ONNX Kp lets the robot sag.
    hold_kp = np.full(d.NUM_ACTIONS, 200.0, dtype=np.float32)
    hold_kd = np.full(d.NUM_ACTIONS, 5.0, dtype=np.float32)
    t_settle = time.time()
    standing = None
    while time.time() - t_settle < 2.0:
        low, sport = io.snapshot()
        if low is None or sport is None:
            time.sleep(0.02)
            continue
        imu_q = np.array(low.imu_state.quaternion, dtype=np.float64)
        imu_q /= max(np.linalg.norm(imu_q), 1e-8)
        gyro = np.array(low.imu_state.gyroscope, dtype=np.float64)
        pelvis_pos, pelvis_quat, _ = imu_to_pelvis(
            np.array(sport.position, dtype=np.float64),
            imu_q,
            np.array(sport.velocity, dtype=np.float64),
            gyro,
        )
        rpy = np.array(low.imu_state.rpy, dtype=np.float64)
        if pelvis_pos[2] > 0.5 and abs(rpy[0]) < 0.5 and abs(rpy[1]) < 0.5:
            standing = (low, sport, pelvis_pos, pelvis_quat)
            break
        time.sleep(0.02)
    if standing is None:
        low0, sport0 = io.snapshot()
        imu_q = np.array(low0.imu_state.quaternion, dtype=np.float64)
        imu_q /= max(np.linalg.norm(imu_q), 1e-8)
        pelvis_pos0, pelvis_quat0, _ = imu_to_pelvis(
            np.array(sport0.position, dtype=np.float64),
            imu_q,
            np.array(sport0.velocity, dtype=np.float64),
            np.array(low0.imu_state.gyroscope, dtype=np.float64),
        )
        rpy = np.array(low0.imu_state.rpy, dtype=np.float64)
        raise SystemExit(
            "Refusing to start policy: robot is not upright "
            f"(pelvis_z={pelvis_pos0[2]:.3f}, roll={rpy[0]:.3f}, pitch={rpy[1]:.3f}). "
            "Reset unitree_mujoco and reconnect while it is waiting for rt/lowcmd."
        )
    else:
        low0, sport0, pelvis_pos0, pelvis_quat0 = standing
    q0, _ = xml_from_lowstate(low0)
    target_start_raw = d.policy_to_xml(
        motion["joint_pos"][start_frame].astype(np.float32), joint_seq
    )
    target_start, start_clip_mask = clip_joint_targets(
        target_start_raw, args.joint_limit_margin
    )
    if np.any(start_clip_mask):
        names = ", ".join(np.asarray(d.JOINT_XML)[start_clip_mask])
        print(f"[WARN] start pose clipped to joint limits: {names}")
    print(
        f"[sdk] preparing motion frame {start_frame} -> pelvis xy={pelvis_pos0[:2]} "
        f"z={pelvis_pos0[2]:.3f} kp_scale={args.policy_kp_scale}"
    )

    cmd_lock = threading.Lock()
    q_cmd, kp_cmd, kd_cmd = pack_full(q0, hold_kp, hold_kd)
    stop = threading.Event()

    def write_loop():
        with cmd_lock:
            q, kp, kd = q_cmd.copy(), kp_cmd.copy(), kd_cmd.copy()
        low, _ = io.snapshot()
        mm = int(low.mode_machine) if low is not None else 4
        io.send(q, kp, kd, mm)

    writer = RecurrentThread(interval=CMD_DT, target=write_loop, name="lowcmd")
    writer.Start()

    timestep = start_frame
    hold_left = HOLD_POLICY_STEPS
    action_buffer = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    action_filtered = np.zeros(d.NUM_ACTIONS, dtype=np.float32)
    obs = np.zeros(d.NUM_OBS, dtype=np.float32)
    motion_pos = None
    motion_quat = None
    policy_aligned = False
    handoff_sync_tick = None
    handoff_fresh_samples = 0
    alpha = 0.8
    max_t = int(motion["joint_pos"].shape[0])
    # Drive the policy from the robot clock rather than the host wall clock.
    # unitree_mujoco can briefly run slower/faster than wall time while rendering;
    # using time.time() here makes the reference phase drift away from physics.
    # LowState.tick is milliseconds on both unitree_mujoco and the real G1.
    start_tick = None
    last_policy_tick = None
    policy_wall_start = None
    policy_start_tick = None
    policy_updates = 0
    skipped_motion_frames = 0
    limit_clip_events = 0
    limit_clip_max = 0.0
    wall_watchdog_start = time.monotonic()
    now = 0.0
    print(f"[beyondmimic] prepare {args.prepare_s:.1f}s then policy {args.duration:.1f}s")
    try:
        while not stop.is_set():
            if time.monotonic() - wall_watchdog_start > args.prepare_s + 3.0 * args.duration + 10.0:
                print("[SAFETY] stopping: robot clock did not complete within wall-time watchdog")
                break
            low, sport = io.snapshot()
            if low is None or sport is None:
                time.sleep(0.001)
                continue

            tick = int(low.tick) & 0xFFFFFFFF
            if start_tick is None:
                start_tick = tick
            elapsed_tick_ms = (tick - start_tick) & 0xFFFFFFFF
            now = elapsed_tick_ms * 1e-3
            if now >= args.prepare_s + args.duration:
                break
            if (
                last_policy_tick is not None
                and ((tick - last_policy_tick) & 0xFFFFFFFF) < POLICY_TICK_MS
            ):
                time.sleep(0.001)
                continue
            last_policy_tick = tick
            q_xml, dq_xml = xml_from_lowstate(low)
            imu_q = np.array(low.imu_state.quaternion, dtype=np.float64)
            imu_q /= max(np.linalg.norm(imu_q), 1e-8)
            gyro = np.array(low.imu_state.gyroscope, dtype=np.float64)
            imu_pos = np.array(sport.position, dtype=np.float64)
            imu_vel = np.array(sport.velocity, dtype=np.float64)
            pelvis_pos, pelvis_quat, pelvis_lin_w = imu_to_pelvis(imu_pos, imu_q, imu_vel, gyro)
            waist = float(q_xml[d.JOINT_XML.index("waist_yaw_joint")])
            torso_pos, torso_quat = pelvis_to_torso(pelvis_pos, pelvis_quat, waist)
            rpy = np.array(low.imu_state.rpy, dtype=np.float64)
            if pelvis_pos[2] < 0.45 or abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0:
                print(
                    "[SAFETY] stopping: robot fell "
                    f"(pelvis_z={pelvis_pos[2]:.3f}, roll={rpy[0]:.3f}, pitch={rpy[1]:.3f})"
                )
                break

            if now < args.prepare_s:
                a = float(np.clip(now / max(args.prepare_s, 1e-3), 0.0, 1.0))
                q_des = (1.0 - a) * q0 + a * target_start
                with cmd_lock:
                    q_cmd[:], kp_cmd[:], kd_cmd[:] = pack_full(q_des, hold_kp, hold_kd)
                continue

            # Finish the static settling phase before aligning the world-frame
            # motion.  Aligning first and then holding for one second lets the
            # unactuated root translate/yaw, so the reference is already stale
            # when the first policy action is applied.
            if hold_left > 0:
                hold_left -= 1
                with cmd_lock:
                    q_cmd[:], kp_cmd[:], kd_cmd[:] = pack_full(
                        target_start, hold_kp, hold_kd
                    )
                continue

            # DDS discovery/startup may initially deliver an old sample followed
            # by a multi-second jump to the live stream. Keep holding the start
            # pose until several consecutive robot-clock deltas are reasonable;
            # only then capture the world-frame alignment used by the policy.
            if not policy_aligned:
                if handoff_sync_tick is None:
                    handoff_sync_tick = tick
                    handoff_fresh_samples = 0
                else:
                    handoff_delta_ms = (tick - handoff_sync_tick) & 0xFFFFFFFF
                    handoff_sync_tick = tick
                    if 0 < handoff_delta_ms <= 60:
                        handoff_fresh_samples += 1
                    elif handoff_delta_ms > 60:
                        handoff_fresh_samples = 0
                if handoff_fresh_samples < 5:
                    with cmd_lock:
                        q_cmd[:], kp_cmd[:], kd_cmd[:] = pack_full(
                            target_start, hold_kp, hold_kd
                        )
                    continue

            # Align at the actual policy handoff, after preparation and settling.
            # The robot can translate/yaw during either phase, so an earlier
            # alignment produces a large anchor observation on the first action.
            # Isaac's tracking anchor is torso_link, therefore align torso-to-torso
            # rather than pelvis-to-pelvis.
            if not policy_aligned:
                motion_pos, motion_quat = align_motion_to_robot(
                    motion["body_pos_w"],
                    motion["body_quat_w"],
                    motion["body_pos_w"][start_frame, d.TORSO_BODY_INDEX],
                    motion["body_quat_w"][start_frame, d.TORSO_BODY_INDEX],
                    torso_pos,
                    torso_quat,
                )
                timestep = start_frame
                hold_left = HOLD_POLICY_STEPS
                action_buffer[:] = 0.0
                action_filtered[:] = 0.0
                obs[:] = 0.0
                policy_aligned = True
                policy_wall_start = time.monotonic()
                policy_start_tick = tick
                policy_updates = 0
                aligned_pos, aligned_quat = d.subtract_frame_transforms_mujoco(
                    torso_pos.astype(np.float32),
                    torso_quat.astype(np.float32),
                    motion_pos[timestep, d.TORSO_BODY_INDEX],
                    motion_quat[timestep, d.TORSO_BODY_INDEX],
                )
                aligned_mat = np.zeros(9)
                mujoco.mju_quat2Mat(aligned_mat, aligned_quat)
                aligned_ori = aligned_mat.reshape(3, 3)[:, :2].reshape(-1)
                print(
                    "[sdk] policy handoff aligned torso "
                    f"frame={timestep} |anchor_pos|={np.linalg.norm(aligned_pos):.4f} "
                    f"anchor_pos={aligned_pos} anchor_ori={aligned_ori}"
                )

            assert motion_pos is not None and motion_quat is not None
            assert policy_start_tick is not None
            scheduled_timestep = start_frame + (
                ((tick - policy_start_tick) & 0xFFFFFFFF) // POLICY_TICK_MS
            )
            if scheduled_timestep - timestep > 5:
                gap = scheduled_timestep - timestep
                motion_pos, motion_quat = align_motion_to_robot(
                    motion["body_pos_w"],
                    motion["body_quat_w"],
                    motion["body_pos_w"][start_frame, d.TORSO_BODY_INDEX],
                    motion["body_quat_w"][start_frame, d.TORSO_BODY_INDEX],
                    torso_pos,
                    torso_quat,
                )
                policy_start_tick = tick
                timestep = start_frame
                action_buffer[:] = 0.0
                action_filtered[:] = 0.0
                scheduled_timestep = start_frame
                print(
                    f"[WARN] robot clock discontinuity ({gap} motion frames); "
                    "re-aligned and restarted from start_frame"
                )
            if scheduled_timestep > timestep:
                skipped_motion_frames += max(0, scheduled_timestep - timestep)
                timestep = min(scheduled_timestep, max_t - 1)
            policy_updates += 1

            motion_cmd = np.concatenate(
                [motion["joint_pos"][timestep], motion["joint_vel"][timestep]]
            ).astype(np.float32)
            anchor_pos, anchor_quat = d.subtract_frame_transforms_mujoco(
                torso_pos.astype(np.float32),
                torso_quat.astype(np.float32),
                motion_pos[timestep, d.TORSO_BODY_INDEX],
                motion_quat[timestep, d.TORSO_BODY_INDEX],
            )
            anchor_mat = np.zeros(9)
            mujoco.mju_quat2Mat(anchor_mat, anchor_quat)
            anchor_ori = anchor_mat.reshape(3, 3)[:, :2].reshape(-1).astype(np.float32)
            base_lin = d.quat_rotate_inverse_np(pelvis_quat, pelvis_lin_w).astype(np.float32)
            gyro_f = gyro.astype(np.float32)
            q_pol = d.xml_to_policy(q_xml, joint_seq)
            dq_pol = d.xml_to_policy(dq_xml, joint_seq)

            o = 0
            obs[o : o + 46] = motion_cmd
            o += 46
            obs[o : o + 3] = anchor_pos
            o += 3
            obs[o : o + 6] = anchor_ori
            o += 6
            obs[o : o + 3] = base_lin
            o += 3
            obs[o : o + 3] = gyro_f
            o += 3
            obs[o : o + d.NUM_ACTIONS] = q_pol - default_seq
            o += d.NUM_ACTIONS
            obs[o : o + d.NUM_ACTIONS] = dq_pol
            o += d.NUM_ACTIONS
            obs[o : o + d.NUM_ACTIONS] = action_buffer

            action = session.run(
                ["actions"],
                {
                    "obs": obs[None, :].astype(np.float32),
                    "time_step": np.array([[float(timestep)]], dtype=np.float32),
                },
            )[0]
            action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -5.0, 5.0)
            action_buffer = action.copy()
            action_filtered = alpha * action + (1.0 - alpha) * action_filtered
            q_des_raw = d.policy_to_xml(
                action_filtered * action_scale_seq + default_seq, joint_seq
            )
            q_des, limit_clip_mask = clip_joint_targets(
                q_des_raw, args.joint_limit_margin
            )
            limit_clip_events += int(np.count_nonzero(limit_clip_mask))
            current_clip_max = float(np.max(np.abs(q_des_raw - q_des)))
            limit_clip_max = max(limit_clip_max, current_clip_max)
            kp_now, kd_now = stiffness_xml, damping_xml
            timestep += 1
            if timestep >= max_t:
                print("[beyondmimic] motion finished")
                break

            with cmd_lock:
                q_cmd[:], kp_cmd[:], kd_cmd[:] = pack_full(q_des, kp_now, kd_now)

            if timestep % 50 == 0:
                q_error = q_des - q_xml
                tau_xml = tau_from_lowstate(low)
                worst = np.argsort(np.abs(q_error))[-3:][::-1]
                worst_text = ",".join(
                    f"{d.JOINT_XML[i]}:{q_error[i]:+.3f}rad/{tau_xml[i]:+.1f}Nm"
                    for i in worst
                )
                wall_elapsed = max(time.monotonic() - policy_wall_start, 1e-6)
                print(
                    f"[INFO] t={timestep} pelvis_z={pelvis_pos[2]:.3f} "
                    f"anchor={np.array2string(anchor_pos, precision=3)} "
                    f"rpy={np.array2string(rpy, precision=3)} "
                    f"policy_hz={policy_updates / wall_elapsed:.1f} "
                    f"skipped={skipped_motion_frames} "
                    f"limit_clips={limit_clip_events} "
                    f"clip_max={limit_clip_max:.3f}rad "
                    f"qerr_rms={np.sqrt(np.mean(q_error * q_error)):.3f} "
                    f"worst=[{worst_text}]"
                )
    except KeyboardInterrupt:
        print("[beyondmimic] interrupted")
    finally:
        print(
            f"[sdk] summary robot_time={now:.3f}s policy_updates={policy_updates} "
            f"motion_frame={timestep} skipped={skipped_motion_frames} "
            f"limit_clips={limit_clip_events} clip_max={limit_clip_max:.3f}rad"
        )
        stop.set()
        with cmd_lock:
            kp_cmd[:] = 0.0
            kd_cmd[:] = 8.0
        time.sleep(0.2)
        writer.Wait(timeout=1.0)
        print("[sdk] stopped")


if __name__ == "__main__":
    main()
