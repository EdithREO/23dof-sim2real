#!/usr/bin/env python3
"""G1 23DoF: swap MuJoCo proprio for unitree_sdk2, send PD targets via LowCmd.

Sim2sim 闭环:
  MuJoCo LowState 等价量 -> 组观测 -> ONNX -> residual action -> PD target q*
  -> tau = kp*(q*-q) - kd*dq -> 写进仿真

真机闭环（本脚本）:
  SDK LowState (电机 q/dq + IMU) -> 同一套观测 -> ONNX -> 同一套 PD target q*
  -> LowCmd: 默认把 q*/kp/kd 交给电机板载 PD（推荐）
     或 --command-mode torque 把算出来的 tau 发给 SDK（kp=kd=0）

先在 unitree_mujoco 上用 --network lo 验证，再上真机。
真机必须加 --allow-real，并关掉机载运控、进入 debug。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT / "tw_cp_mujoco_119" / "scripts"))
from evaluation_contract import (  # noqa: E402
    BASELINE_ACTION_FILTER_ALPHA,
    BASELINE_DELAY_STEPS,
    CANONICAL_ACTION_SCALE,
    CANONICAL_KD,
    CANONICAL_KP,
    CANONICAL_POLICY_JOINT_ORDER,
)

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

G1_NUM_MOTOR = 29
# unitree_hg LowCmd/LowState motor index == g1_23dof.xml <actuator> order.
# Policy (21) skips waist_roll/pitch (13/14) and all wrist pitch/yaw; locks wrist_roll.
MOTOR_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
POLICY_TO_MOTOR = (
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18,
    22, 23, 24, 25,
)
# 23DoF 有腕 roll，策略没有；锁在 0 位。
HOLD_MOTORS = (19, 26)  # left_wrist_roll, right_wrist_roll
G1_23DOF_MOTORS = tuple(sorted(set(POLICY_TO_MOTOR) | set(HOLD_MOTORS)))
assert len(POLICY_TO_MOTOR) == 21
assert HOLD_MOTORS == (19, 26)
assert MOTOR_NAMES[19] == "left_wrist_roll" and MOTOR_NAMES[26] == "right_wrist_roll"
assert all(MOTOR_NAMES[i] == n for i, n in zip(
    POLICY_TO_MOTOR,
    (
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    ),
))
assert tuple(CANONICAL_POLICY_JOINT_ORDER) == (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
)

DEPLOY_DEFAULT = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0,
        0.0, 0.0, 0.0, 0.87,
        0.0, 0.0, 0.0, 0.87,
    ],
    dtype=np.float32,
)
ANKLE_IDX = (4, 5, 10, 11)
N_ACT = 21
N_MIMIC = 9 + N_ACT + 6  # current + future step 5
N_PROPRIO = 3 + 2 + 2 + 3 * N_ACT
N_OBS_SINGLE = N_MIMIC + N_PROPRIO
HISTORY_LEN = 10
N_POLICY_OBS = N_OBS_SINGLE * (HISTORY_LEN + 1)
POLICY_DT = 0.02
CMD_DT = 0.002
DECIMATION = int(round(POLICY_DT / CMD_DT))

OBS_ANG_VEL_SCALE = 0.25
OBS_DOF_POS_SCALE = 1.0
OBS_DOF_VEL_SCALE = 0.05
CLIP_OBS = 100.0
CLIP_ACTIONS = 0.6
FUTURE_STEP = 5
MIMIC_FIRST_STEP = 1


def wrap_to_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2 * np.pi) - np.pi


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q = [w,x,y,z], rotate v into body frame."""
    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class MotionRef:
    def __init__(self, csv_path: Path, fps: float):
        df = pd.read_csv(csv_path)
        self.fps = float(fps)
        n = len(df)
        self.t = np.arange(n, dtype=np.float64) / self.fps
        self.root_z = df["root_z"].to_numpy(np.float64)
        self.roll = df["roll"].to_numpy(np.float64)
        self.pitch = df["pitch"].to_numpy(np.float64)
        self.yaw = df["yaw"].to_numpy(np.float64)
        self.vel = df[["root_vel_x", "root_vel_y", "root_vel_z"]].to_numpy(np.float64)
        self.ang = df[["root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z"]].to_numpy(np.float64)
        self.dof = df[[f"dof_pos_{i}" for i in range(N_ACT)]].to_numpy(np.float64)
        qw = df["root_quat_w"].to_numpy(np.float64)
        qx = df["root_quat_x"].to_numpy(np.float64)
        qy = df["root_quat_y"].to_numpy(np.float64)
        qz = df["root_quat_z"].to_numpy(np.float64)
        self.quat_wxyz = np.stack([qw, qx, qy, qz], axis=1)
        self.duration = float(self.t[-1]) if n else 0.0

    def _interp(self, t: float) -> dict:
        t = float(np.clip(t, 0.0, self.duration))
        x = self.t
        def s(arr):
            if arr.ndim == 1:
                return float(np.interp(t, x, arr))
            return np.stack([np.interp(t, x, arr[:, i]) for i in range(arr.shape[1])], axis=0)

        return {
            "root_z": s(self.root_z),
            "roll": s(self.roll),
            "pitch": s(self.pitch),
            "yaw": s(self.yaw),
            "vel": s(self.vel),
            "ang": s(self.ang),
            "dof": s(self.dof).astype(np.float32),
            "quat": s(self.quat_wxyz),
        }

    def mimic_and_ref_dof(self, t: float, robot_yaw: float) -> tuple[np.ndarray, np.ndarray]:
        cur = self._interp(t + MIMIC_FIRST_STEP * POLICY_DT)
        fut = self._interp(t + FUTURE_STEP * POLICY_DT)
        heading = wrap_to_pi(cur["yaw"] - robot_yaw)
        fut_heading = wrap_to_pi(fut["yaw"] - robot_yaw)
        vel_local = quat_rotate_inverse_wxyz(cur["quat"], cur["vel"])
        ang_local = quat_rotate_inverse_wxyz(cur["quat"], cur["ang"])
        fut_vel_local = quat_rotate_inverse_wxyz(fut["quat"], fut["vel"])
        fut_ang_local = quat_rotate_inverse_wxyz(fut["quat"], fut["ang"])
        mimic = np.concatenate(
            [
                np.array([cur["root_z"], cur["roll"], cur["pitch"], np.sin(heading), np.cos(heading)], dtype=np.float32),
                vel_local.astype(np.float32),
                np.array([ang_local[2]], dtype=np.float32),
                cur["dof"],
                fut_vel_local.astype(np.float32),
                np.array([np.sin(fut_heading), np.cos(fut_heading), fut_ang_local[2]], dtype=np.float32),
            ]
        )
        if mimic.shape[0] != N_MIMIC:
            raise ValueError(f"mimic dim {mimic.shape[0]} != {N_MIMIC}")
        return mimic.astype(np.float32), cur["dof"].astype(np.float32)


class G1SdkIO:
    def __init__(self, publish: bool):
        self.publish = publish
        self._lock = threading.Lock()
        self.low_state = None
        self.mode_machine = 4
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_pub.Init()
        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self._on_state, 10)

    def _on_state(self, msg: LowState_):
        with self._lock:
            self.low_state = msg
            self.mode_machine = int(msg.mode_machine)

    def wait_state(self, timeout_s: float = 5.0) -> LowState_:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self.low_state is not None:
                    return self.low_state
            time.sleep(0.01)
        raise TimeoutError("未收到 rt/lowstate。检查网卡、DDS、是否已关运控 / unitree_mujoco 是否在跑。")

    def read_proprio(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            msg = self.low_state
        if msg is None:
            raise RuntimeError("low_state is empty")
        q = np.zeros(N_ACT, dtype=np.float32)
        dq = np.zeros(N_ACT, dtype=np.float32)
        for i, mid in enumerate(POLICY_TO_MOTOR):
            q[i] = float(msg.motor_state[mid].q)
            dq[i] = float(msg.motor_state[mid].dq)
        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        rpy = np.array(msg.imu_state.rpy, dtype=np.float32)
        return q, dq, gyro, rpy

    def send(self, q_full: np.ndarray, dq_full: np.ndarray, kp_full: np.ndarray, kd_full: np.ndarray, tau_full: np.ndarray):
        with self._lock:
            mode_machine = self.mode_machine
        cmd = self.low_cmd
        cmd.mode_pr = 0
        cmd.mode_machine = mode_machine
        for i in range(G1_NUM_MOTOR):
            m = cmd.motor_cmd[i]
            if i in G1_23DOF_MOTORS:
                m.mode = 1
                m.q = float(q_full[i])
                m.dq = float(dq_full[i])
                m.kp = float(kp_full[i])
                m.kd = float(kd_full[i])
                m.tau = float(tau_full[i])
            else:
                m.mode = 0
                m.q = 0.0
                m.dq = 0.0
                m.kp = 0.0
                m.kd = 0.0
                m.tau = 0.0
        cmd.crc = self.crc.Crc(cmd)
        if self.publish:
            self.lowcmd_pub.Write(cmd)


class PolicyRuntime:
    def __init__(self, onnx_path: Path, motion: MotionRef):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        if tuple(d if isinstance(d, int) else N_POLICY_OBS for d in inp.shape)[-1] not in (N_POLICY_OBS, None):
            print(f"[warn] ONNX input shape {inp.shape}, expected last dim {N_POLICY_OBS}")
        self.input_name = inp.name
        self.motion = motion
        self.action_scale = np.asarray(CANONICAL_ACTION_SCALE, dtype=np.float32)
        self.kp = np.asarray(CANONICAL_KP, dtype=np.float32)
        self.kd = np.asarray(CANONICAL_KD, dtype=np.float32)
        self.obs_hist = np.zeros((HISTORY_LEN, N_OBS_SINGLE), dtype=np.float32)
        self.filtered = np.zeros(N_ACT, dtype=np.float32)
        self.action_buf = np.zeros((5, N_ACT), dtype=np.float32)
        self.last_action = np.zeros(N_ACT, dtype=np.float32)
        self.step = 0
        self.ref_dof = DEPLOY_DEFAULT.copy()

    def _proprio(self, q, dq, gyro, rpy, first: bool) -> np.ndarray:
        roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
        act_hist = np.zeros(N_ACT, dtype=np.float32) if first else self.last_action
        dq_obs = dq.copy()
        for i in ANKLE_IDX:
            dq_obs[i] = 0.0
        proprio = np.concatenate(
            [
                gyro * OBS_ANG_VEL_SCALE,
                np.array([roll, pitch, np.sin(yaw), np.cos(yaw)], dtype=np.float32),
                (q - DEPLOY_DEFAULT) * OBS_DOF_POS_SCALE,
                dq_obs * OBS_DOF_VEL_SCALE,
                act_hist,
            ]
        )
        if proprio.shape[0] != N_PROPRIO:
            raise ValueError(f"proprio dim {proprio.shape[0]} != {N_PROPRIO}")
        return proprio.astype(np.float32), yaw

    def infer(self, t: float, q, dq, gyro, rpy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first = self.step <= 1
        proprio, yaw = self._proprio(q, dq, gyro, rpy, first)
        mimic, self.ref_dof = self.motion.mimic_and_ref_dof(t, yaw)
        single = np.concatenate([mimic, proprio]).astype(np.float32)
        if first:
            self.obs_hist[:] = single
        policy_obs = np.concatenate([single, self.obs_hist.reshape(-1)])
        policy_obs = np.clip(policy_obs, -CLIP_OBS, CLIP_OBS)
        raw = self.session.run(None, {self.input_name: policy_obs[None, :]})[0].reshape(-1).astype(np.float32)
        clip_lim = CLIP_ACTIONS / self.action_scale
        a = np.clip(raw, -clip_lim, clip_lim)
        alpha = BASELINE_ACTION_FILTER_ALPHA
        a = (1.0 - alpha) * self.filtered + alpha * a
        self.filtered = a
        self.action_buf[:-1] = self.action_buf[1:]
        self.action_buf[-1] = a
        delayed = self.action_buf[-(BASELINE_DELAY_STEPS + 1)].copy()
        self.last_action = delayed
        q_des = self.ref_dof + delayed * self.action_scale
        tau = (q_des - q) * self.kp - dq * self.kd
        self.obs_hist[:-1] = self.obs_hist[1:]
        self.obs_hist[-1] = single
        self.step += 1
        return delayed, q_des, tau


def pack_full(policy_q: np.ndarray, policy_kp: np.ndarray, policy_kd: np.ndarray, policy_tau: np.ndarray):
    """Pack 21 policy joints into 29-motor LowCmd; lock both wrist_roll at 0."""
    if policy_q.shape[0] != N_ACT:
        raise ValueError(f"policy_q dim {policy_q.shape[0]} != {N_ACT}")
    q = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    dq = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    kp = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    kd = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    tau = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
    for i, mid in enumerate(POLICY_TO_MOTOR):
        q[mid] = policy_q[i]
        kp[mid] = policy_kp[i]
        kd[mid] = policy_kd[i]
        tau[mid] = policy_tau[i]
    # Lock left/right wrist_roll (motors 19, 26); unused dummy motors stay kp=0.
    for mid in HOLD_MOTORS:
        q[mid] = 0.0
        dq[mid] = 0.0
        kp[mid] = 20.0
        kd[mid] = 1.0
        tau[mid] = 0.0
    return q, dq, kp, kd, tau


def print_motor_map() -> None:
    print("[sdk] LowCmd motor map (policy 21 → unitree_hg 29; wrist_roll locked):")
    for i, mid in enumerate(POLICY_TO_MOTOR):
        print(f"  policy[{i:2d}] {CANONICAL_POLICY_JOINT_ORDER[i]:28s} -> motor[{mid:2d}] {MOTOR_NAMES[mid]}")
    for mid in HOLD_MOTORS:
        print(f"  LOCK                         -> motor[{mid:2d}] {MOTOR_NAMES[mid]} q=0")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 23DoF SDK sim2real for model_24500 RMG policy")
    p.add_argument("--network", default="lo", help="DDS 网卡。仿真用 lo，真机用 enpXs0")
    p.add_argument("--onnx", type=Path, default=BUNDLE_ROOT / "policy" / "model_24500_actor.onnx")
    p.add_argument("--motion-csv", type=Path, default=BUNDLE_ROOT / "motion" / "neutral_held_50hz.csv")
    p.add_argument("--csv-fps", type=float, default=50.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--command-mode", choices=("position", "torque"), default="position")
    p.add_argument("--dry-run", action="store_true", help="只读 SDK 状态并推理，不发 LowCmd")
    p.add_argument("--allow-real", action="store_true", help="允许对非 lo 网卡发指令（真机）")
    p.add_argument("--prepare-s", type=float, default=3.0, help="先 PD 锁 CSV 首帧姿态，再交给策略")
    p.add_argument(
        "--kp-scale",
        type=float,
        default=None,
        help="同时缩放发给 SDK 的 kp/kd（unitree_mujoco 动力学偏软，lo 上默认 8）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.network != "lo" and not args.dry_run and not args.allow_real:
        raise SystemExit("真机发指令需要同时指定 --allow-real。建议先 --network lo 对接 unitree_mujoco。")

    cyclonedds_home = Path("/home/liuboqian_/Project/23dof_sim2real/cyclonedds/install")
    if cyclonedds_home.is_dir():
        os.environ["CYCLONEDDS_HOME"] = str(cyclonedds_home)

    ChannelFactoryInitialize(0, args.network)
    publish = not args.dry_run
    io = G1SdkIO(publish=publish)
    print(f"[sdk] waiting lowstate on {args.network} ...")
    io.wait_state()
    print(f"[sdk] got lowstate, mode_machine={io.mode_machine}")
    print_motor_map()

    motion = MotionRef(args.motion_csv, args.csv_fps)
    policy = PolicyRuntime(args.onnx, motion)
    start_q = motion._interp(0.0)["dof"].astype(np.float32)
    kp_scale = float(args.kp_scale) if args.kp_scale is not None else (8.0 if args.network == "lo" else 1.0)
    policy.kp = policy.kp * kp_scale
    policy.kd = policy.kd * kp_scale
    print(
        f"[sdk] kp_scale={kp_scale} (kd scaled same; wrist_roll locked at motors 19/26; "
        f"prepare holds CSV frame-0 for {args.prepare_s:.1f}s, q0={start_q[0]:.3f})"
    )
    cmd_lock = threading.Lock()
    q_cmd, dq_cmd, kp_cmd, kd_cmd, tau_cmd = pack_full(
        start_q, policy.kp, policy.kd, np.zeros(N_ACT)
    )

    stop = threading.Event()

    def write_loop():
        with cmd_lock:
            q, dq, kp, kd, tau = q_cmd.copy(), dq_cmd.copy(), kp_cmd.copy(), kd_cmd.copy(), tau_cmd.copy()
        io.send(q, dq, kp, kd, tau)

    writer = RecurrentThread(interval=CMD_DT, target=write_loop, name="lowcmd")
    if publish:
        writer.Start()

    t0 = time.time()
    last_policy = 0.0
    print(
        f"[sim2real] mode={args.command_mode} publish={publish} duration={args.duration}s "
        f"onnx={args.onnx.name}"
    )
    try:
        while not stop.is_set():
            now = time.time() - t0
            if now >= args.duration:
                break
            if now - last_policy < POLICY_DT:
                time.sleep(0.001)
                continue
            last_policy = now
            q, dq, gyro, rpy = io.read_proprio()
            if now < args.prepare_s:
                # Hold CSV frame-0 so policy handoff does not yank from squat to ref.
                q_des = start_q.copy()
                tau = (q_des - q) * policy.kp - dq * policy.kd
            else:
                t_motion = now - args.prepare_s
                _, q_des, tau = policy.infer(t_motion, q, dq, gyro, rpy)

            if args.command_mode == "position":
                tau_send = np.zeros_like(tau)
                kp_send, kd_send = policy.kp, policy.kd
                q_send = q_des
            else:
                q_send = q
                kp_send = np.zeros(N_ACT, dtype=np.float32)
                kd_send = np.zeros(N_ACT, dtype=np.float32)
                tau_send = tau

            packed = pack_full(q_send, kp_send, kd_send, tau_send)
            with cmd_lock:
                q_cmd, dq_cmd, kp_cmd, kd_cmd, tau_cmd = packed
            if int(now * 50) % 25 == 0:
                print(f"t={now:5.2f}s roll={rpy[0]:6.3f} pitch={rpy[1]:6.3f} q0={q[0]:6.3f} qdes0={q_des[0]:6.3f}")
    except KeyboardInterrupt:
        print("[sim2real] interrupted, damping...")
    finally:
        stop.set()
        # 退出时把 kp 降下来，避免突然卸力。
        q, dq, gyro, rpy = io.read_proprio()
        packed = pack_full(q, np.zeros(N_ACT), policy.kd * 0.5, np.zeros(N_ACT))
        with cmd_lock:
            q_cmd, dq_cmd, kp_cmd, kd_cmd, tau_cmd = packed
        time.sleep(0.2)
        if publish:
            writer.Wait()
        print("[sim2real] stopped")


if __name__ == "__main__":
    main()
