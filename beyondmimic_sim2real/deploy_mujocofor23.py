"""MuJoCo sim2sim for BeyondMimic G1 23DoF.

Default obs layout matches Tracking-Flat-G123-Wo-State-Estimation-v0 (124-dim):
  command(46) + motion_anchor_ori_b(6) + base_ang_vel(3)
  + joint_pos(23) + joint_vel(23) + actions(23)
i.e. no motion_anchor_pos_b / base_lin_vel (unavailable on real LowState-only deploy).

Why it could not stand:
  1) Spawn near XML origin while motion starts at pelvis~(1.5, 4.5, 0.8)
     -> anchor observations explode.
  2) g1_23dof.xml enables mesh self-collision and almost no joint armature
     -> PD cannot hold a pose.
  3) Feet often spawn a few cm above the floor -> impact then collapse.

Fix: spawn at motion frame (+ velocities), disable mesh collisions, set armature,
settle feet, scale policy Kp, delay/lowpass actions, fix free-joint ang-vel obs.

Standby: PD-hold the ONNX default pose until you press Space, then snap to
motion frame 0 and start the tracking policy.

Run:
  conda activate unitree_rl_mjlab
  cd beyondmimic_sim2real
  python deploy_mujocofor23.py
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime

np.set_printoptions(precision=6, suppress=True)

XML_PATH = "./unitree_description/mjcf/g1_23dof.xml"
MOTION_PATH = "./g1_play/motion.npz"
POLICY_PATH = "./g1_play/policy.onnx"

SIMULATION_DURATION = 300.0
SIMULATION_DT = 0.002
CONTROL_DECIMATION = 10  # 50 Hz
HOLD_STEPS = 50
STANDBY_KP_SCALE = 8.0  # ONNX Kp is too soft to hold a stand in this MJCF
POLICY_KP_SCALE = 4.0  # same scale: Isaac Implicit PD Kp is too soft as MuJoCo motors
ACTION_DELAY = 0  # control steps; matches common Unitree deploy delay
ACTION_LOWPASS_ALPHA = 0.8  # y = a*x + (1-a)*y_prev
KEY_SPACE = 32
KEY_R = 82
KEY_P = 80

NUM_ACTIONS = 23
# Wo-State-Estimation policy obs (no anchor_pos / base_lin_vel).
NUM_OBS = 124
TORSO_BODY_INDEX = 3  # full npz: pelvis=0, torso_link=3

ARMATURE_BY_JOINT = {
    "left_hip_pitch_joint": 0.01017752004,
    "right_hip_pitch_joint": 0.01017752004,
    "left_hip_roll_joint": 0.025101925,
    "right_hip_roll_joint": 0.025101925,
    "left_hip_yaw_joint": 0.01017752004,
    "right_hip_yaw_joint": 0.01017752004,
    "left_knee_joint": 0.025101925,
    "right_knee_joint": 0.025101925,
    "left_ankle_pitch_joint": 0.00721945,
    "right_ankle_pitch_joint": 0.00721945,
    "left_ankle_roll_joint": 0.00721945,
    "right_ankle_roll_joint": 0.00721945,
    "waist_yaw_joint": 0.01017752004,
    "left_shoulder_pitch_joint": 0.003609725,
    "right_shoulder_pitch_joint": 0.003609725,
    "left_shoulder_roll_joint": 0.003609725,
    "right_shoulder_roll_joint": 0.003609725,
    "left_shoulder_yaw_joint": 0.003609725,
    "right_shoulder_yaw_joint": 0.003609725,
    "left_elbow_joint": 0.003609725,
    "right_elbow_joint": 0.003609725,
    "left_wrist_roll_joint": 0.003609725,
    "right_wrist_roll_joint": 0.003609725,
}

JOINT_XML = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    dot = np.sum(q_vec * v, axis=-1, keepdims=True)
    c = q_vec * dot * 2.0
    return a - b + c


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):
    rotm_a = np.zeros(9)
    mujoco.mju_quat2Mat(rotm_a, quat_a)
    rotm_a = rotm_a.reshape(3, 3)
    rel_pos = rotm_a.T @ (pos_b - pos_a)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    return rel_pos.astype(np.float32), rel_quat.astype(np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def policy_to_xml(values: np.ndarray, joint_seq: list[str]) -> np.ndarray:
    return np.array([values[joint_seq.index(j)] for j in JOINT_XML], dtype=np.float32)


def xml_to_policy(values: np.ndarray, joint_seq: list[str]) -> np.ndarray:
    return np.array([values[JOINT_XML.index(j)] for j in joint_seq], dtype=np.float32)


def load_onnx_metadata(path: str):
    model = onnx.load(path)
    meta = {p.key: p.value for p in model.metadata_props}
    need = ["joint_names", "default_joint_pos", "joint_stiffness", "joint_damping", "action_scale"]
    missing = [k for k in need if k not in meta]
    if missing:
        raise KeyError(f"ONNX metadata missing {missing}; have {sorted(meta)}")

    joint_seq = meta["joint_names"].split(",")
    default_seq = np.asarray([float(x) for x in meta["default_joint_pos"].split(",")], dtype=np.float32)
    stiffness_seq = np.asarray([float(x) for x in meta["joint_stiffness"].split(",")], dtype=np.float32)
    damping_seq = np.asarray([float(x) for x in meta["joint_damping"].split(",")], dtype=np.float32)
    action_scale_seq = np.asarray([float(x) for x in meta["action_scale"].split(",")], dtype=np.float32)
    print("[INFO] anchor:", meta.get("anchor_body") or meta.get("anchor_body_name"))
    print("[INFO] obs:", meta.get("observation_names"))
    return joint_seq, default_seq, stiffness_seq, damping_seq, action_scale_seq


def prepare_model(model: mujoco.MjModel) -> None:
    mesh_disabled = 0
    for i in range(model.ngeom):
        if int(model.geom_type[i]) == 7:  # mesh
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0
            mesh_disabled += 1

    armature_set = 0
    for name, arm in ARMATURE_BY_JOINT.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        dofid = int(model.jnt_dofadr[jid])
        model.dof_armature[dofid] = arm
        if model.dof_frictionloss[dofid] == 0.0:
            model.dof_frictionloss[dofid] = 0.1
        armature_set += 1

    print(f"[INFO] disabled mesh collision on {mesh_disabled} geoms; set armature on {armature_set} joints")


def settle_feet_to_ground(model: mujoco.MjModel, data: mujoco.MjData, clearance: float = 0.002) -> None:
    mujoco.mj_forward(model, data)
    foot_ids = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "foot" in name and model.geom_contype[i] != 0:
            foot_ids.append(i)

    if not foot_ids:
        for bname in ("left_ankle_roll_link", "right_ankle_roll_link"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                z = float(data.xpos[bid][2])
                if z > 0.03:
                    data.qpos[2] -= z - 0.03
        mujoco.mj_forward(model, data)
        print(f"[INFO] settle fallback pelvis z={data.qpos[2]:.3f}")
        return

    def min_foot_z() -> float:
        mujoco.mj_forward(model, data)
        return float(min(data.geom_xpos[i][2] - model.geom_size[i][0] for i in foot_ids))

    z = min_foot_z()
    if z > clearance:
        data.qpos[2] -= z - clearance
        mujoco.mj_forward(model, data)
    print(f"[INFO] foot clearance after settle: {min_foot_z():.4f} m, pelvis z={data.qpos[2]:.3f}")


def clip_tau(model: mujoco.MjModel, tau: np.ndarray) -> np.ndarray:
    out = tau.copy()
    for j, name in enumerate(JOINT_XML):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_actfrcrange[jid]
        if hi > 0:
            out[j] = np.clip(out[j], lo, hi)
    return out


def spawn_standby(model, data, default_xml: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qpos[2] = 0.76
    data.qpos[3] = 1.0
    data.qpos[7 : 7 + NUM_ACTIONS] = default_xml
    data.qvel[:] = 0.0
    settle_feet_to_ground(model, data)
    print("[INFO] STANDBY: default stand pose. Press Space/P to start policy, R to reset standby.")


def spawn_motion0(
    model,
    data,
    motion_pos_w,
    motion_quat_w,
    motion_joint_pos,
    motion_joint_vel,
    motion_body_lin_vel_w,
    motion_body_ang_vel_w,
    joint_seq,
    frame: int = 0,
) -> None:
    """Teleport to a motion frame, including root/joint velocities.

    MuJoCo free-joint angular qvel is expressed in the body frame, while the
    npz stores world-frame body angular velocity.
    """
    root_quat = motion_quat_w[frame, 0].astype(np.float64)
    data.qpos[0:3] = motion_pos_w[frame, 0]
    data.qpos[3:7] = root_quat
    data.qpos[7 : 7 + NUM_ACTIONS] = policy_to_xml(
        motion_joint_pos[frame].astype(np.float32), joint_seq
    )
    data.qvel[:] = 0.0
    settle_feet_to_ground(model, data)

    # Re-apply velocities after settle (settle only adjusts z).
    lin_w = motion_body_lin_vel_w[frame, 0].astype(np.float64)
    ang_w = motion_body_ang_vel_w[frame, 0].astype(np.float64)
    ang_b = quat_rotate_inverse_np(data.qpos[3:7].astype(np.float64), ang_w)
    data.qvel[0:3] = lin_w
    data.qvel[3:6] = ang_b
    data.qvel[6 : 6 + NUM_ACTIONS] = policy_to_xml(
        motion_joint_vel[frame].astype(np.float32), joint_seq
    )
    mujoco.mj_forward(model, data)
    print(
        f"[INFO] POLICY: spawned at motion frame {frame} "
        f"|lin|={np.linalg.norm(lin_w):.3f} |ang_b|={np.linalg.norm(ang_b):.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="G1 23DoF MuJoCo sim2sim")
    parser.add_argument(
        "--start_mode",
        choices=("standby", "policy"),
        default="standby",
        help="standby = PD hold default pose until Space; policy = start tracking immediately",
    )
    parser.add_argument(
        "--policy_kp_scale",
        type=float,
        default=POLICY_KP_SCALE,
        help="multiply ONNX Kp/Kd for MuJoCo motors in policy mode",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=10,
        help="motion frame used when entering policy mode (skip retarget glitch at 0)",
    )
    parser.add_argument(
        "--action_delay",
        type=int,
        default=ACTION_DELAY,
        help="delay applied actions by this many control steps",
    )
    parser.add_argument(
        "--action_lowpass",
        type=float,
        default=ACTION_LOWPASS_ALPHA,
        help="low-pass alpha on actions; 1=no filter",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    motion = np.load(MOTION_PATH)
    motion_pos_w = motion["body_pos_w"]
    motion_quat_w = motion["body_quat_w"]
    motion_joint_pos = motion["joint_pos"]
    motion_joint_vel = motion["joint_vel"]
    motion_body_lin_vel_w = motion["body_lin_vel_w"]
    motion_body_ang_vel_w = motion["body_ang_vel_w"]
    max_timestep = int(motion_pos_w.shape[0])
    start_frame = int(np.clip(args.start_frame, 0, max_timestep - 1))
    print(
        f"[INFO] motion frames={max_timestep} joints={motion_joint_pos.shape[1]} "
        f"bodies={motion_pos_w.shape[1]}"
    )
    print(
        f"[INFO] frame0 pelvis={motion_pos_w[0, 0]} torso={motion_pos_w[0, TORSO_BODY_INDEX]} "
        f"start_frame={start_frame}"
    )

    joint_seq, default_seq, stiffness_seq, damping_seq, action_scale_seq = load_onnx_metadata(POLICY_PATH)
    stiffness_xml = policy_to_xml(stiffness_seq, joint_seq)
    damping_xml = policy_to_xml(damping_seq, joint_seq)
    default_xml = policy_to_xml(default_seq, joint_seq)
    standby_kp = stiffness_xml * STANDBY_KP_SCALE
    standby_kd = damping_xml * (STANDBY_KP_SCALE**0.5)
    policy_kp = stiffness_xml * float(args.policy_kp_scale)
    policy_kd = damping_xml * (float(args.policy_kp_scale) ** 0.5)
    print(
        f"[INFO] policy_kp_scale={args.policy_kp_scale} "
        f"action_delay={args.action_delay} action_lowpass={args.action_lowpass}"
    )

    session = onnxruntime.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in session.get_inputs()]
    print("[INFO] ONNX inputs:", in_names)
    if "obs" not in in_names or "time_step" not in in_names:
        raise RuntimeError(f"unexpected ONNX inputs: {in_names}")
    obs_dim = int(session.get_inputs()[in_names.index("obs")].shape[-1])
    if obs_dim != NUM_OBS:
        raise RuntimeError(
            f"ONNX obs dim {obs_dim} != deploy NUM_OBS {NUM_OBS}; "
            "packing must match observation_names (Wo-SE 124)."
        )
    print(f"[INFO] obs dim aligned: {obs_dim}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_DT
    prepare_model(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if body_id < 0:
        raise ValueError("body torso_link not found in MJCF")

    mode = {"name": args.start_mode}
    timestep = start_frame if args.start_mode == "policy" else 0
    hold_left = HOLD_STEPS
    action_buffer = np.zeros(NUM_ACTIONS, dtype=np.float32)
    action_filtered = np.zeros(NUM_ACTIONS, dtype=np.float32)
    delay_steps = max(0, int(args.action_delay))
    action_queue = [np.zeros(NUM_ACTIONS, dtype=np.float32) for _ in range(delay_steps)]
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    counter = 0
    alpha = float(np.clip(args.action_lowpass, 0.0, 1.0))

    def enter_policy() -> None:
        nonlocal timestep, hold_left, action_buffer, action_filtered, action_queue, target_dof_pos, counter
        spawn_motion0(
            model,
            data,
            motion_pos_w,
            motion_quat_w,
            motion_joint_pos,
            motion_joint_vel,
            motion_body_lin_vel_w,
            motion_body_ang_vel_w,
            joint_seq,
            frame=start_frame,
        )
        timestep = start_frame
        hold_left = HOLD_STEPS
        action_buffer = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_filtered = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_queue = [np.zeros(NUM_ACTIONS, dtype=np.float32) for _ in range(delay_steps)]
        target_dof_pos = data.qpos[7 : 7 + NUM_ACTIONS].astype(np.float32).copy()
        counter = 0
        mode["name"] = "policy"

    if args.start_mode == "standby":
        spawn_standby(model, data, default_xml)
        target_dof_pos = default_xml.copy()
    else:
        enter_policy()

    def key_callback(keycode: int) -> None:
        nonlocal timestep, hold_left, action_buffer, action_filtered, action_queue, target_dof_pos, counter
        if keycode in (KEY_SPACE, KEY_P):
            if mode["name"] != "policy":
                enter_policy()
                print("[INFO] switched STANDBY -> POLICY (Space/P)")
        elif keycode == KEY_R:
            spawn_standby(model, data, default_xml)
            timestep = 0
            hold_left = HOLD_STEPS
            action_buffer = np.zeros(NUM_ACTIONS, dtype=np.float32)
            action_filtered = np.zeros(NUM_ACTIONS, dtype=np.float32)
            action_queue = [np.zeros(NUM_ACTIONS, dtype=np.float32) for _ in range(delay_steps)]
            target_dof_pos = default_xml.copy()
            counter = 0
            mode["name"] = "standby"
            print("[INFO] reset to STANDBY (R)")

    print("[INFO] keys: Space/P = start policy, R = standby reset")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < SIMULATION_DURATION:
            step_start = time.time()
            if mode["name"] == "policy" and timestep >= max_timestep:
                print("[INFO] motion finished")
                break

            if mode["name"] == "standby":
                kp, kd, tgt = standby_kp, standby_kd, default_xml
            else:
                kp, kd, tgt = policy_kp, policy_kd, target_dof_pos

            tau = pd_control(
                tgt,
                data.qpos[7 : 7 + NUM_ACTIONS],
                kp,
                np.zeros(NUM_ACTIONS, dtype=np.float32),
                data.qvel[6 : 6 + NUM_ACTIONS],
                kd,
            )
            data.ctrl[:] = clip_tau(model, tau)
            mujoco.mj_step(model, data)
            counter += 1

            if mode["name"] == "policy" and counter % CONTROL_DECIMATION == 0:
                motion_cmd = np.concatenate(
                    [motion_joint_pos[timestep], motion_joint_vel[timestep]]
                ).astype(np.float32)

                anchor_pos, anchor_quat = subtract_frame_transforms_mujoco(
                    data.xpos[body_id],
                    data.xquat[body_id],
                    motion_pos_w[timestep, TORSO_BODY_INDEX],
                    motion_quat_w[timestep, TORSO_BODY_INDEX],
                )
                anchor_mat = np.zeros(9)
                mujoco.mju_quat2Mat(anchor_mat, anchor_quat)
                anchor_ori = anchor_mat.reshape(3, 3)[:, :2].reshape(-1).astype(np.float32)

                root_quat = data.qpos[3:7].copy()
                # free-joint: lin vel is world-frame, ang vel is already body-frame
                base_lin_vel = quat_rotate_inverse_np(root_quat, data.qvel[0:3]).astype(np.float32)
                base_ang_vel = data.qvel[3:6].astype(np.float32)

                q_policy = xml_to_policy(data.qpos[7 : 7 + NUM_ACTIONS].astype(np.float32), joint_seq)
                dq_policy = xml_to_policy(data.qvel[6 : 6 + NUM_ACTIONS].astype(np.float32), joint_seq)

                o = 0
                obs[o : o + 46] = motion_cmd
                o += 46
                obs[o : o + 6] = anchor_ori
                o += 6
                obs[o : o + 3] = base_ang_vel
                o += 3
                obs[o : o + NUM_ACTIONS] = q_policy - default_seq
                o += NUM_ACTIONS
                obs[o : o + NUM_ACTIONS] = dq_policy
                o += NUM_ACTIONS
                obs[o : o + NUM_ACTIONS] = action_buffer
                o += NUM_ACTIONS
                if o != NUM_OBS:
                    raise RuntimeError(f"obs pack length {o} != NUM_OBS {NUM_OBS}")
                action = session.run(
                    ["actions"],
                    {
                        "obs": obs[None, :].astype(np.float32),
                        "time_step": np.array([[float(timestep)]], dtype=np.float32),
                    },
                )[0]
                action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -5.0, 5.0)

                if hold_left > 0:
                    # Hold: freeze motion index and PD-track the reference pose so the
                    # robot settles with matching velocities before closed-loop tracking.
                    hold_left -= 1
                    action_buffer = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    action_filtered = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    action_queue = [np.zeros(NUM_ACTIONS, dtype=np.float32) for _ in range(delay_steps)]
                    target_dof_pos = policy_to_xml(
                        motion_joint_pos[timestep].astype(np.float32), joint_seq
                    )
                else:
                    action_buffer = action.copy()
                    if delay_steps > 0:
                        action_queue.append(action.copy())
                        delayed = action_queue.pop(0)
                    else:
                        delayed = action
                    action_filtered = alpha * delayed + (1.0 - alpha) * action_filtered
                    target_dof_pos = policy_to_xml(
                        action_filtered * action_scale_seq + default_seq, joint_seq
                    )
                    timestep += 1

                if timestep % 50 == 0:
                    print(
                        f"[INFO] t={timestep} z={data.qpos[2]:.3f} "
                        f"|anchor|={np.linalg.norm(anchor_pos):.3f}"
                    )
            elif mode["name"] == "standby" and counter % 250 == 0:
                print(f"[INFO] STANDBY z={data.qpos[2]:.3f}  (Space/P -> policy)")

            viewer.sync()
            sleep_t = model.opt.timestep - (time.time() - step_start)
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
