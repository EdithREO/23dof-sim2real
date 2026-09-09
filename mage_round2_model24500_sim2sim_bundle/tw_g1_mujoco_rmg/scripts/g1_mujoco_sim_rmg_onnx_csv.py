#!/usr/bin/env python3
"""
MuJoCo sim2sim：G1 21-DoF ONNX Actor + CSV 参考轨迹。

MAGE80250 locked evaluation contract：baseline 固定 action delay=1、低通
alpha=0.5、motor tau=0，并在 motion end 后执行 10 s terminal hold/cosine velocity decay。
候选控制只允许 delay {0,1,2} × alpha {0.4,0.5,0.6}；contract-bound run 禁止 --v4。
验证 g1_rmg_v4_s2s_repair checkpoint 时仍可显式使用 --v4（clip_actions=0.5、无
delay/低通、balance-safe target_offset_clip + joint clamp、root_reset_z_offset=-0.017），
但它不属于 MAGE80250 contract。
诊断可用 --pure-ref-pd / --zero-policy-action。

实现复用 tw_cp_mujoco_119/scripts/c1_mujoco_sim_cp_rmg_onnx_csv.py，仅覆盖 G1 关节名、
PD、默认姿态、action scale、delay/低通与 MuJoCo 足端/终止体名称。

机器人常量来源：legged_gym/legged_gym/envs/c_p/g1_23dof_rmg_config.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PRJ_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _PRJ_DIR.parent
# 优先加载与 g1 脚本同目录的 c1_mujoco_sim_cp_rmg_onnx_csv.py
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_CP_SCRIPTS = _REPO_ROOT / "tw_cp_mujoco_119" / "scripts"
if _CP_SCRIPTS.is_dir() and str(_CP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_CP_SCRIPTS))

import c1_mujoco_sim_cp_rmg_onnx_csv as cp  # noqa: E402
import jitter_slip_diagnostics as jitter_diag_module  # noqa: E402
from jitter_slip_diagnostics import (  # noqa: E402
    JitterSlipDiagnostics,
    JitterSlipDiagnosticsConfig,
    _euler_xyz_from_quat_xyzw,
)
from evaluation_contract import (  # noqa: E402
    BASELINE_ACTION_FILTER_ALPHA,
    BASELINE_DELAY_STEPS,
    BASELINE_MOTOR_TAU_S,
    CANONICAL_ABI,
    CANONICAL_ANKLE_VELOCITY_MASK_INDICES,
    CANONICAL_ACTION_SCALE,
    CANONICAL_EFFORT_LIMITS,
    CANONICAL_KD,
    CANONICAL_KP,
    CANONICAL_OBSERVATION_FIELDS,
    CANONICAL_OBSERVATION_SCALES,
    CANONICAL_POLICY_JOINT_ORDER,
    CANONICAL_VELOCITY_LIMITS,
    CONTRACT_SCHEMA,
    POST_MOTION_HOLD_S,
    canonical_abi,
    file_identity,
    inspect_xml_topology,
    sha256_file,
    validate_canonical_abi,
)
from g1_policy_in_loop_ablation import (  # noqa: E402
    ControlSweepConfig,
    ControlTraceRecorder,
    ObsAblationState,
    apply_group_action_scale,
    build_enhanced_summary,
    build_joint_group_indices,
    load_control_trace,
    parse_ablate_action_groups,
    parse_obs_ablation,
)

_DEFAULT_XML = _REPO_ROOT / "assets" / "unitree_g1_23dof" / "g1_21dof.xml"
_DEFAULT_ONNX = _PRJ_DIR / "config" / "onnx" / "model_13000_actor.onnx"
_DEFAULT_MOTION_CSV = _PRJ_DIR / "config" / "g1_cmu_0527_20" / "105_105_15_stageii.csv"
_G1_URDF = _REPO_ROOT / "assets" / "unitree_g1_23dof" / "g1_21dof.urdf"

# 与 legged_gym/envs/c_p/g1_23dof_rmg_config.py motion.key_bodies / upper_key_bodies 一致
G1_KEY_BODIES: tuple[str, ...] = (
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_elbow_link",
    "left_shoulder_yaw_link",
    "right_elbow_link",
    "right_shoulder_yaw_link",
)
G1_UPPER_KEY_BODIES: tuple[str, ...] = (
    "torso_link",
    "left_elbow_link",
    "left_shoulder_yaw_link",
    "right_elbow_link",
    "right_shoulder_yaw_link",
)

G1_POLICY_DOF_NAMES: tuple[str, ...] = tuple(CANONICAL_POLICY_JOINT_ORDER)

G1_STUDENT327_FUTURE_REF_STEPS: tuple[int, ...] = (5,)
G1_STUDENT327_DEPLOY_DEFAULT_DOF_POS: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0,
    0.0, 0.0, 0.0, 0.87,
    0.0, 0.0, 0.0, 0.87,
)
G1_STUDENT327_N_MIMIC_OBS = 9 + len(G1_POLICY_DOF_NAMES) + 6 * len(
    G1_STUDENT327_FUTURE_REF_STEPS
)
G1_STUDENT327_N_PROPRIO_OBS = 3 + 2 + 2 + 3 * len(G1_POLICY_DOF_NAMES)
G1_STUDENT327_N_OBS_SINGLE = G1_STUDENT327_N_MIMIC_OBS + G1_STUDENT327_N_PROPRIO_OBS
G1_STUDENT327_N_POLICY_OBS = G1_STUDENT327_N_OBS_SINGLE * (cp.HISTORY_LEN + 1)


def build_g1_student327_single_obs_labels() -> list[str]:
    """G1 student327 actor obs labels, aligned with C306RMGStudent327ObsDistillV1Cfg."""
    labels: list[str] = [
        "mimic/root_height",
        "mimic/ref_roll",
        "mimic/ref_pitch",
        "mimic/ref_heading_sin",
        "mimic/ref_heading_cos",
        "mimic/ref_vel_x",
        "mimic/ref_vel_y",
        "mimic/ref_vel_z",
        "mimic/ref_ang_vel_yaw",
    ]
    labels.extend(f"mimic/ref_dof_pos_{i}" for i in range(len(G1_POLICY_DOF_NAMES)))
    for step in G1_STUDENT327_FUTURE_REF_STEPS:
        labels.extend(
            (
                f"mimic/future_{step}/ref_vel_x",
                f"mimic/future_{step}/ref_vel_y",
                f"mimic/future_{step}/ref_vel_z",
                f"mimic/future_{step}/ref_heading_sin",
                f"mimic/future_{step}/ref_heading_cos",
                f"mimic/future_{step}/ref_ang_vel_yaw",
            )
        )
    labels.extend(
        (
            "proprio/base_ang_vel_x",
            "proprio/base_ang_vel_y",
            "proprio/base_ang_vel_z",
            "proprio/roll",
            "proprio/pitch",
            "proprio/yaw_sin",
            "proprio/yaw_cos",
        )
    )
    labels.extend(f"proprio/dof_pos_minus_deploy_default_{i}" for i in range(len(G1_POLICY_DOF_NAMES)))
    labels.extend(f"proprio/dof_vel_{i}" for i in range(len(G1_POLICY_DOF_NAMES)))
    labels.extend(f"proprio/deploy_action_history_{i}" for i in range(len(G1_POLICY_DOF_NAMES)))
    assert len(labels) == G1_STUDENT327_N_OBS_SINGLE, (
        f"label count {len(labels)} != G1_STUDENT327_N_OBS_SINGLE "
        f"{G1_STUDENT327_N_OBS_SINGLE}"
    )
    return labels


def _validate_g1_student327_deploy_alignment(
    cfg: cp.RMGConfig,
    onnx_obs_dim: int,
    onnx_has_norm: bool | None,
) -> None:
    if cfg.history_len != cp.HISTORY_LEN:
        raise ValueError(f"history_len must be {cp.HISTORY_LEN}, got {cfg.history_len}")
    if onnx_obs_dim != G1_STUDENT327_N_POLICY_OBS:
        raise ValueError(
            f"ONNX input dim must be {G1_STUDENT327_N_POLICY_OBS}, got {onnx_obs_dim}"
        )
    if abs(float(cfg.clip_observations) - 100.0) > 1e-6:
        raise ValueError(
            "clip_observations must be 100.0 for training/ONNX alignment, "
            f"got {cfg.clip_observations}"
        )
    if cfg.action_delay_steps is not None:
        delay_desc = str(cfg.action_delay_steps)
    elif cfg.action_delay:
        bounds = cp.CP_RMG_V1.get("ctrl_delay_step_bounds", None)
        if bounds is not None:
            delay_desc = f"rand {bounds[0]}..{bounds[1]}"
        else:
            delay_desc = f"rand 0..{cfg.ctrl_delay_step_range}"
    else:
        delay_desc = "off"
    if onnx_has_norm is True:
        norm_desc = "embedded"
    elif onnx_has_norm is False:
        norm_desc = "none(student327 normalize_obs=False)"
    else:
        norm_desc = "unknown"
    action_scale = np.asarray(cfg.action_scale, dtype=np.float64)
    norm_clip_max = float(np.max(float(cfg.clip_actions) / action_scale))
    max_res = cp._max_residual_rad(float(cfg.clip_actions), action_scale)
    print(
        "[deploy:g1] obs: raw "
        f"{G1_STUDENT327_N_POLICY_OBS} = {G1_STUDENT327_N_OBS_SINGLE}×(1+{cfg.history_len}); "
        "mimic=student327 current step 1 + future step 5 heading(ref_yaw-robot_yaw); "
        "proprio=ang_vel+roll/pitch+yaw_sin/cos+dof/default+vel+deploy_action_history; "
        f"clip_obs=100 before ONNX; ONNX normalizer={norm_desc}; "
        f"actions: residual_clip={cfg.clip_actions} rad "
        f"(norm_clip_max={norm_clip_max:.4f}, action_scale_max={float(np.max(action_scale)):.4f}, "
        f"effective_max_residual={max_res:.4f} rad); "
        f"lowpass={cfg.action_lowpass_filter}(alpha={cfg.action_filter_alpha}) "
        f"delay={cfg.action_delay}(steps={delay_desc}) "
        f"balance_safe={getattr(cfg, 'g1_balance_safe_enable', False)} "
        f"yaw_align_to_ref={cfg.yaw_align_to_ref}",
        flush=True,
    )

# G123DoFRMGCfg.control / init_state / asset / normalization（与 train_rmg_g1.sh 一致）
_G1_ACTION_SCALE = tuple(CANONICAL_ACTION_SCALE)

# G123DoFRMGRealRepairCfg.domain_rand.joint_armature（policy dof 顺序，与 G1_POLICY_DOF_NAMES 一致）
_G1_JOINT_ARMATURE = np.asarray(
    [
        0.010177520, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450,
        0.010177520, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450,
        0.010177520,
        0.003609725, 0.003609725, 0.003609725, 0.003609725,
        0.003609725, 0.003609725, 0.003609725, 0.003609725,
    ],
    dtype=np.float32,
)

# Locked training-side limits from
# legged_gym/envs/c_p/g1_23dof_rmg_config.py:G123DoFRMGCfg.control.
# Keep these in policy order; the URDF is used only for position limits.
_G1_TRAINING_EFFORT_LIMITS = np.asarray(CANONICAL_EFFORT_LIMITS, dtype=np.float32)
_G1_TRAINING_VELOCITY_LIMITS = np.asarray(CANONICAL_VELOCITY_LIMITS, dtype=np.float32)
_G1_TRAINING_LIMIT_SOURCE = (
    "legged_gym/legged_gym/envs/c_p/g1_23dof_rmg_config.py"
    ":G123DoFRMGCfg.control"
)

G1_RMG_V1: dict = {
    **cp.CP_RMG_V1,
    "default_joint_angles": {
        "left_hip_pitch_joint": -0.312,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.312,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.669,
        "right_ankle_pitch_joint": -0.363,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.87,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.87,
    },
    "action_scale": _G1_ACTION_SCALE,
    "stiffness": {
        "hip_pitch": 40.179238,
        "hip_roll": 99.098428,
        "hip_yaw": 40.179238,
        "knee": 99.098428,
        "ankle": 28.501246,
        "waist": 40.179238,
        "shoulder_pitch": 14.250623,
        "shoulder_roll": 14.250623,
        "shoulder_yaw": 14.250623,
        "elbow": 14.250623,
    },
    "damping": {
        "hip_pitch": 2.557890,
        "hip_roll": 6.308802,
        "hip_yaw": 2.557890,
        "knee": 6.308802,
        "ankle": 1.814446,
        "waist": 2.557890,
        "shoulder_pitch": 0.907223,
        "shoulder_roll": 0.907223,
        "shoulder_yaw": 0.907223,
        "elbow": 0.907223,
    },
    "clip_actions": 0.6,
    "urdf_limit_discount": 1.0,
    # G123DoFMAGE80250ModerateDRContinueCfg.domain_rand: deployment uses the
    # midpoint of the trained low-pass range and samples only trained delays.
    "action_buf_len": 3,
    "ctrl_delay_step_bounds": (0, 2),
    "ctrl_delay_step_range": 2,
    "action_filter_alpha": 0.50,
    "action_filter_alpha_range": (0.35, 0.65),
    "torque_safety_limit": 1.0,
    "enable_tn_limit": False,
    "root_reset_z_offset": 0.05,
    "motion_termination_hold_time": 10.0,
    "motion_termination_vel_decay": "cosine",
    "key_bodies": G1_KEY_BODIES,
    "upper_key_bodies": G1_UPPER_KEY_BODIES,
    "ankle_idx": (4, 5, 10, 11),
    "terminate_after_contacts_on": ("pelvis", "torso_link"),
    "urdf_basename": "g1_21dof.urdf",
    "ankle_2_ground": 0.05,
    # Legacy v4 reset only. MAGE80250 uses root_reset_z_offset=+0.05 exactly.
    "initial_sole_clearance_m": 0.005,
}

# G123DoFRMGBalanceSafeStartupV4S2SRepairCfg.control（与 g1_rmg_v4_s2s_repair 部署一致）
_G1_V4_S2S_TARGET_OFFSET_CLIP = (
    0.42, 0.14, 0.16, 0.48, 0.24, 0.05,
    0.42, 0.14, 0.16, 0.48, 0.24, 0.05,
    0.08,
    0.10, 0.10, 0.10, 0.10,
    0.10, 0.10, 0.10, 0.10,
)

G1_RMG_V4: dict = {
    **G1_RMG_V1,
    "clip_actions": 0.50,
    "action_delay": False,
    "action_lowpass_filter": False,
    "action_filter_alpha": 1.0,
    "root_reset_z_offset": -0.017,
    # cp_fdd_g1._compute_g1_balance_safe_pd_targets
    "g1_balance_safe_enable": True,
    "enable_target_offset_clip": True,
    "enable_target_joint_limit_clamp": True,
    "target_joint_limit_margin": 0.005,
    "target_offset_clip": _G1_V4_S2S_TARGET_OFFSET_CLIP,
}

_G1_CFG_EXTRA_KEYS = (
    "g1_balance_safe_enable",
    "enable_target_offset_clip",
    "enable_target_joint_limit_clamp",
    "target_joint_limit_margin",
    "target_offset_clip",
    "pure_ref_pd",
    "motion_termination_hold_time",
    "motion_termination_vel_decay",
)


def _apply_g1_patches(
    *,
    use_v4: bool = False,
    ablation: ControlSweepConfig | None = None,
) -> None:
    ablation = ablation or ControlSweepConfig()
    base_cfg = G1_RMG_V4 if use_v4 else G1_RMG_V1
    cp.POLICY_DOF_NAMES = G1_POLICY_DOF_NAMES
    cp.N_MIMIC_OBS = G1_STUDENT327_N_MIMIC_OBS
    cp.N_TRACKING_ERROR_OBS = 0
    cp.N_UPPER_BODY_OBS = 0
    cp.N_PROPRIO_OBS = G1_STUDENT327_N_PROPRIO_OBS
    cp.N_OBS_SINGLE = G1_STUDENT327_N_OBS_SINGLE
    cp.N_POLICY_OBS = G1_STUDENT327_N_POLICY_OBS
    cp.build_cp_fdd_v1_single_obs_labels = build_g1_student327_single_obs_labels
    cp._validate_deploy_alignment = _validate_g1_student327_deploy_alignment
    cp.CP_RMG_V1 = base_cfg
    cp.CP_RMG_V1_CLEAN = {**base_cfg, **{k: v for k, v in cp.CP_RMG_V1_CLEAN.items() if k not in base_cfg}}
    cp._FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

    def _estimate_ref_feet_contact_prob(dof_pos_ref: np.ndarray, dof_names: tuple) -> np.ndarray:
        lap = float(dof_pos_ref[dof_names.index("left_ankle_pitch_joint")])
        rap = float(dof_pos_ref[dof_names.index("right_ankle_pitch_joint")])

        def _prob(ap: float) -> float:
            return 0.001 if ap > cp._REF_SWING_ANKLE_PITCH_THRESH else 0.999

        return np.array([_prob(lap), _prob(rap)], dtype=np.float64)

    cp._estimate_ref_feet_contact_prob = _estimate_ref_feet_contact_prob

    def _g1_csv_motion_lib_init(self, csv_file, device, fps=50.0):
        self._device = device
        self._body_link_list = list(G1_KEY_BODIES)
        self._load_csv(csv_file, fps)

    cp.CsvMotionLib.__init__ = _g1_csv_motion_lib_init

    def _g1_build_body_mappings(self):
        body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)
        ]
        name_to_id = {n: i for i, n in enumerate(body_names)}
        missing_upper = [n for n in self.cfg.upper_key_bodies if n not in name_to_id]
        if missing_upper:
            raise ValueError(f"MuJoCo model missing upper key bodies: {missing_upper}")
        missing_key = [n for n in G1_KEY_BODIES if n not in name_to_id]
        if missing_key:
            raise ValueError(f"MuJoCo model missing key bodies: {missing_key}")

    cp.MujocoRMGSim._build_body_mappings = _g1_build_body_mappings

    def _g1_build_motor_limits(self):
        if tuple(self.cfg.dof_names) != G1_POLICY_DOF_NAMES:
            raise ValueError(
                "G1 locked limit contract requires the canonical 21-DoF policy order"
            )
        n = int(self.cfg.num_actions)
        if n != len(G1_POLICY_DOF_NAMES):
            raise ValueError(f"G1 locked limit contract requires 21 actions, got {n}")
        if _G1_TRAINING_EFFORT_LIMITS.size != n or _G1_TRAINING_VELOCITY_LIMITS.size != n:
            raise ValueError("G1 locked limit arrays do not match num_actions")
        if not np.all(np.isfinite(_G1_TRAINING_EFFORT_LIMITS)) or np.any(
            _G1_TRAINING_EFFORT_LIMITS <= 0.0
        ):
            raise ValueError("G1 training effort limits must be finite and positive")
        if not np.all(np.isfinite(_G1_TRAINING_VELOCITY_LIMITS)) or np.any(
            _G1_TRAINING_VELOCITY_LIMITS <= 0.0
        ):
            raise ValueError("G1 training velocity limits must be finite and positive")
        _, _, pos_limits_policy = self._load_urdf_joint_limits()
        if pos_limits_policy is None:
            raise ValueError("G1 position limits could not be loaded from the configured URDF")
        safety = float(self.cfg.torque_safety_limit)
        discount = float(self.cfg.urdf_limit_discount)
        if not np.isfinite(safety) or safety <= 0.0:
            raise ValueError(f"invalid torque_safety_limit={safety}")
        if not np.isfinite(discount) or discount <= 0.0:
            raise ValueError(f"invalid urdf_limit_discount={discount}")
        effort_act = self._policy_to_act_vec(_G1_TRAINING_EFFORT_LIMITS)
        velocity_act = self._policy_to_act_vec(_G1_TRAINING_VELOCITY_LIMITS)
        self.torque_limits = effort_act * safety
        self.motor_tau_stall = effort_act * discount
        self.motor_omega_nl = velocity_act * discount
        self.dof_pos_limits_policy = pos_limits_policy
        self._g1_training_effort_limits_policy = _G1_TRAINING_EFFORT_LIMITS.copy()
        self._g1_training_velocity_limits_policy = _G1_TRAINING_VELOCITY_LIMITS.copy()
        self._g1_limit_contract_metadata = {
            "source": _G1_TRAINING_LIMIT_SOURCE,
            "joint_names_policy_order": list(G1_POLICY_DOF_NAMES),
            "training_effort_limits_policy": [
                float(value) for value in _G1_TRAINING_EFFORT_LIMITS
            ],
            "training_velocity_limits_policy": [
                float(value) for value in _G1_TRAINING_VELOCITY_LIMITS
            ],
            "torque_safety_limit": safety,
            "urdf_limit_discount": discount,
            "effective_torque_limits_actuator_order": [
                float(value) for value in self.torque_limits
            ],
            "effective_velocity_limits_actuator_order": [
                float(value) for value in self.motor_omega_nl
            ],
            "position_limits_source": os.path.abspath(self.cfg.urdf_path),
        }

    cp.MujocoRMGSim._build_motor_limits = _g1_build_motor_limits

    def _build_dof_mappings(self):
        act_joint_names, act_joint_ids = [], []
        for a in range(self.model.nu):
            jid = int(self.model.actuator_trnid[a, 0])
            act_joint_ids.append(jid)
            act_joint_names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid))
        policy_to_act = [act_joint_names.index(n) for n in self.cfg.dof_names]
        act_to_policy = np.empty((self.cfg.num_actions,), dtype=np.int32)
        for pi, ai in enumerate(policy_to_act):
            act_to_policy[ai] = pi
        self._policy_to_act = np.asarray(policy_to_act, dtype=np.int32)
        self._act_to_policy = act_to_policy
        self._act_qposadr = np.asarray(
            [int(self.model.jnt_qposadr[j]) for j in act_joint_ids], dtype=np.int32
        )
        self._act_qveladr = np.asarray(
            [int(self.model.jnt_dofadr[j]) for j in act_joint_ids], dtype=np.int32
        )
        self.kps_act = self.kps_policy[self._act_to_policy].copy()
        self.kds_act = self.kds_policy[self._act_to_policy].copy()
        self._build_motor_limits()
        ab = getattr(self, "_g1_ablation", None)
        if ab is not None:
            self._g1_apply_pd_scales(ab)
        self.waist_yaw_policy_idx = self.cfg.dof_names.index("waist_yaw_joint")
        self.left_ankle_pitch_policy_idx = self.cfg.dof_names.index("left_ankle_pitch_joint")
        self.right_ankle_pitch_policy_idx = self.cfg.dof_names.index("right_ankle_pitch_joint")
        upper_tokens = ("waist", "shoulder", "elbow", "wrist", "arm")
        upper_body_idx = [
            i for i, n in enumerate(self.cfg.dof_names) if any(tok in n for tok in upper_tokens)
        ]
        self._upper_body_dof_idx = np.asarray(upper_body_idx, dtype=np.int64)
        self._upper_key_body_dim = 3 * len(self.cfg.upper_key_bodies)
        self._build_g1_joint_limit_arrays()

    cp.MujocoRMGSim._build_dof_mappings = _build_dof_mappings

    def _build_g1_joint_limit_arrays(self):
        lo, hi = [], []
        for jname in self.cfg.dof_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"MuJoCo model missing joint: {jname}")
            lo.append(float(self.model.jnt_range[jid, 0]))
            hi.append(float(self.model.jnt_range[jid, 1]))
        self._q_lower_policy = np.asarray(lo, dtype=np.float32)
        self._q_upper_policy = np.asarray(hi, dtype=np.float32)
        toc = getattr(self.cfg, "target_offset_clip", None)
        n = self.cfg.num_actions
        if toc is not None:
            toc_arr = np.asarray(toc, dtype=np.float32).reshape(-1)
            if toc_arr.size != n:
                raise ValueError(
                    f"target_offset_clip length ({toc_arr.size}) != num_actions ({n})"
                )
            self._target_offset_clip = toc_arr
        else:
            self._target_offset_clip = np.full((n,), np.inf, dtype=np.float32)

    cp.MujocoRMGSim._build_g1_joint_limit_arrays = _build_g1_joint_limit_arrays

    def _compute_g1_pd_targets_detailed(self, action_cmd: np.ndarray) -> dict:
        """对齐 cp_fdd_g1._compute_g1_balance_safe_pd_targets；返回中间 target。"""
        action = np.asarray(action_cmd, dtype=np.float32).reshape(-1)
        target_raw = self.ref_dof_pos_res + action * self.action_scale
        target_after_offset_clip = target_raw.copy()
        if not bool(getattr(self.cfg, "g1_balance_safe_enable", False)):
            target = target_raw.copy()
        else:
            target_after_offset_clip = target_raw.copy()
            if bool(getattr(self.cfg, "enable_target_offset_clip", False)):
                offset_raw = target_raw - self.ref_dof_pos_res
                offset = np.clip(
                    offset_raw,
                    -self._target_offset_clip,
                    self._target_offset_clip,
                )
                target_after_offset_clip = self.ref_dof_pos_res + offset
            if bool(getattr(self.cfg, "enable_target_joint_limit_clamp", False)):
                margin = float(getattr(self.cfg, "target_joint_limit_margin", 0.0))
                lower = self._q_lower_policy + margin
                upper = self._q_upper_policy - margin
                invalid = lower > upper
                if np.any(invalid):
                    mid = 0.5 * (self._q_lower_policy + self._q_upper_policy)
                    lower = np.where(invalid, mid, lower)
                    upper = np.where(invalid, mid, upper)
                target = np.clip(target_after_offset_clip, lower, upper)
            else:
                target = target_after_offset_clip.copy()
        target = target.astype(np.float32, copy=True)
        target[self.left_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_left)
        target[self.right_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_right)
        return {
            "target_raw": target_raw.astype(np.float32),
            "target_after_offset_clip": target_after_offset_clip.astype(np.float32),
            "target_after_joint_limit": target,
        }

    def _compute_g1_pd_targets(self, action_cmd: np.ndarray) -> np.ndarray:
        return self._compute_g1_pd_targets_detailed(action_cmd)["target_after_joint_limit"]

    cp.MujocoRMGSim._compute_g1_pd_targets = _compute_g1_pd_targets
    cp.MujocoRMGSim._compute_g1_pd_targets_detailed = _compute_g1_pd_targets_detailed

    def _g1_apply_pd_scales(self, ab: ControlSweepConfig) -> None:
        gi = self._g1_group_indices
        self.kps_act[:] = self.kps_policy[self._act_to_policy]
        self.kds_act[:] = self.kds_policy[self._act_to_policy]
        self.kps_act *= float(ab.kp_scale)
        self.kds_act *= float(ab.kd_scale)
        for pi in gi["ankle"]:
            ai = int(self._policy_to_act[pi])
            self.kps_act[ai] *= float(ab.ankle_kp_scale)
            self.kds_act[ai] *= float(ab.ankle_kd_scale)
        for pi in gi["hip_roll"]:
            ai = int(self._policy_to_act[pi])
            self.kps_act[ai] *= float(ab.hip_roll_kp_scale)
            self.kds_act[ai] *= float(ab.hip_roll_kd_scale)
        if float(ab.torque_limit_scale) != 1.0:
            self.torque_limits = self.torque_limits * float(ab.torque_limit_scale)
        limit_meta = getattr(self, "_g1_limit_contract_metadata", None)
        if limit_meta is not None:
            limit_meta["effective_torque_limits_actuator_order"] = [
                float(value) for value in self.torque_limits
            ]
            limit_meta["torque_limit_scale"] = float(ab.torque_limit_scale)

    cp.MujocoRMGSim._g1_apply_pd_scales = _g1_apply_pd_scales

    def _g1_limit_torques_with_filter(self, tau_raw, dof_vel_act, ab: ControlSweepConfig):
        if float(ab.motor_tau_s) <= 0.0:
            tau_post = self._limit_torques(tau_raw, dof_vel_act)
            return tau_post, tau_post
        control_dt = float(self.cfg.simulation_dt)
        alpha = control_dt / (float(ab.motor_tau_s) + control_dt)
        if not hasattr(self, "_g1_motor_tau_state"):
            self._g1_motor_tau_state = np.zeros_like(tau_raw)
        filtered = self._g1_motor_tau_state + alpha * (tau_raw - self._g1_motor_tau_state)
        self._g1_motor_tau_state = filtered.copy()
        tau_post = self._limit_torques(filtered, dof_vel_act)
        return tau_post, tau_post

    cp.MujocoRMGSim._g1_limit_torques_with_filter = _g1_limit_torques_with_filter

    def _g1_get_base_ang_vel_local(self, _base_quat) -> torch.Tensor:
        # MuJoCo free-joint rotational qvel is already expressed in the body
        # frame. Isaac/MJLab exposes world angular velocity and rotates it into
        # this same frame before building proprioception.
        return torch.tensor(
            self.data.qvel[3:6].copy(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    cp.MujocoRMGSim._get_base_ang_vel_local = _g1_get_base_ang_vel_local

    def _g1_student327_deploy_default_t(self) -> torch.Tensor:
        cached = getattr(self, "_g1_student327_deploy_default_tensor", None)
        if cached is not None:
            return cached
        default = torch.tensor(
            G1_STUDENT327_DEPLOY_DEFAULT_DOF_POS,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        if default.shape[1] != self.cfg.num_actions:
            raise ValueError(
                f"G1 student327 deploy default length {default.shape[1]} "
                f"!= num_actions {self.cfg.num_actions}"
            )
        self._g1_student327_deploy_default_tensor = default
        return default

    cp.MujocoRMGSim._g1_student327_deploy_default_t = _g1_student327_deploy_default_t

    def _g1_student327_obs_action_history(self) -> torch.Tensor:
        # Training's default student327_obs_action_history_chain is "action":
        # expose the clipped/filtered/delayed command exactly as applied.
        deploy_actions = self.action_tensor
        first_step = (self.episode_length_buf <= 1).view(-1, 1)
        return torch.where(first_step, torch.zeros_like(deploy_actions), deploy_actions)

    cp.MujocoRMGSim._g1_student327_obs_action_history = _g1_student327_obs_action_history

    def _g1_sample_student327_ref_features(
        self,
        motion_time: torch.Tensor,
        step: int,
        robot_yaw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        raw_t = motion_time + float(step) * self.dt
        max_t = torch.tensor(
            [max(float(self.motion_len), 0.0)],
            device=self.device,
            dtype=torch.float32,
        )
        query_t = torch.minimum(raw_t, max_t)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _ = self._calc_motion_frame(
            self.motion_ids, raw_t
        )
        # The base runner applies the configured terminal hold velocity decay
        # from the unclamped time; keep the same scaled values for all G1
        # reference consumers instead of applying the decay twice here.
        roll, pitch, ref_yaw = cp.euler_from_quaternion(root_rot)
        heading_delta = cp.wrap_to_pi_torch(ref_yaw - robot_yaw)
        heading_sin = torch.sin(heading_delta).unsqueeze(-1)
        heading_cos = torch.cos(heading_delta).unsqueeze(-1)
        ref_vel_local = cp.quat_rotate_inverse(root_rot, root_vel)
        ref_ang_local = cp.quat_rotate_inverse(root_rot, root_ang_vel)
        return (
            raw_t,
            root_pos,
            roll.unsqueeze(-1),
            pitch.unsqueeze(-1),
            heading_sin,
            heading_cos,
            ref_vel_local,
            ref_ang_local[:, 2:3],
            dof_pos,
        )

    cp.MujocoRMGSim._g1_sample_student327_ref_features = _g1_sample_student327_ref_features

    def _g1_build_student327_mimic_obs(
        self,
        motion_time: torch.Tensor,
        q_policy_t: torch.Tensor,
    ) -> torch.Tensor:
        _, _, robot_yaw = cp.euler_from_quaternion(q_policy_t)
        (
            _cur_raw_t,
            root_pos,
            roll,
            pitch,
            heading_sin,
            heading_cos,
            ref_vel_local,
            ref_ang_yaw,
            dof_pos,
        ) = self._g1_sample_student327_ref_features(
            motion_time,
            int(self.cfg.mimic_tar_first_step),
            robot_yaw,
        )
        self.ref_dof_pos_res = dof_pos[0].detach().cpu().numpy().astype(np.float32)
        cur_feat = torch.cat(
            (
                root_pos[:, 2:3],
                roll,
                pitch,
                heading_sin,
                heading_cos,
                ref_vel_local,
                ref_ang_yaw,
                dof_pos,
            ),
            dim=-1,
        )
        future_feats = []
        motion_len = torch.tensor(
            [float(self.motion_len)], device=self.device, dtype=torch.float32
        )
        cur_future_fallback = torch.cat(
            (ref_vel_local, heading_sin, heading_cos, ref_ang_yaw), dim=-1
        )
        for step in G1_STUDENT327_FUTURE_REF_STEPS:
            (
                raw_t,
                _future_root_pos,
                _future_roll,
                _future_pitch,
                future_heading_sin,
                future_heading_cos,
                future_ref_vel_local,
                future_ref_ang_yaw,
                _future_dof_pos,
            ) = self._g1_sample_student327_ref_features(motion_time, step, robot_yaw)
            future_feat = torch.cat(
                (
                    future_ref_vel_local,
                    future_heading_sin,
                    future_heading_cos,
                    future_ref_ang_yaw,
                ),
                dim=-1,
            )
            invalid = (raw_t >= motion_len).view(-1, 1)
            future_feats.append(torch.where(invalid, cur_future_fallback, future_feat))
        mimic_obs = (
            torch.cat([cur_feat] + future_feats, dim=-1)
            if future_feats
            else cur_feat
        )
        if mimic_obs.shape[-1] != cp.N_MIMIC_OBS:
            raise ValueError(f"Expected G1 student327 mimic dim {cp.N_MIMIC_OBS}, got {mimic_obs.shape[-1]}")
        return mimic_obs

    cp.MujocoRMGSim._g1_build_student327_mimic_obs = _g1_build_student327_mimic_obs

    def _g1_build_student327_actor_obs(self, motion_time: torch.Tensor) -> torch.Tensor:
        _quat_gravity, quat_ang_vel, _q_raw, q_policy = self._obs_orientation_tensors()
        q_policy_t = torch.tensor(
            q_policy, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        base_ang_vel = self._get_base_ang_vel_local(quat_ang_vel)
        roll, pitch, yaw = cp.euler_from_quaternion(q_policy_t)
        observed_roll_pitch = torch.stack((roll, pitch), dim=1)
        yaw_obs = torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=1)
        dof_pos_act, dof_vel_act = self._get_dof_state_act()
        dof_pos_t = torch.tensor(
            self._act_to_policy_vec(dof_pos_act),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        dof_vel_t = torch.tensor(
            self._act_to_policy_vec(dof_vel_act),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        deploy_default = self._g1_student327_deploy_default_t()
        obs_actions = self._g1_student327_obs_action_history()
        proprio_current = torch.cat(
            (
                base_ang_vel * self.cfg.obs_ang_vel_scale,
                observed_roll_pitch,
                yaw_obs,
                (dof_pos_t - deploy_default) * self.cfg.obs_dof_pos_scale,
                dof_vel_t * self.cfg.obs_dof_vel_scale,
                obs_actions,
            ),
            dim=-1,
        )
        if proprio_current.shape[-1] != cp.N_PROPRIO_OBS:
            raise ValueError(
                f"Expected G1 student327 proprio dim {cp.N_PROPRIO_OBS}, "
                f"got {proprio_current.shape[-1]}"
            )
        dof_vel_start_dim = 3 + 2 + 2 + self.cfg.num_actions
        for idx in self.cfg.ankle_idx:
            proprio_current[:, dof_vel_start_dim + idx] = 0.0

        mimic_obs = self._g1_build_student327_mimic_obs(motion_time, q_policy_t)
        student_obs = torch.cat((mimic_obs, proprio_current), dim=-1)
        if student_obs.shape[-1] != cp.N_OBS_SINGLE:
            raise ValueError(
                f"Expected G1 student327 obs dim {cp.N_OBS_SINGLE}, "
                f"got {student_obs.shape[-1]}"
            )
        policy_obs = torch.cat((student_obs, self.obs_history_buf.reshape(1, -1)), dim=-1)
        if policy_obs.shape[-1] != cp.N_POLICY_OBS:
            raise ValueError(
                f"Expected G1 student327 policy obs dim {cp.N_POLICY_OBS}, "
                f"got {policy_obs.shape[-1]}"
            )
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([student_obs] * self.cfg.history_len, dim=1),
            torch.cat((self.obs_history_buf[:, 1:], student_obs.unsqueeze(1)), dim=1),
        )
        return policy_obs

    _orig_build_actor_obs = _g1_build_student327_actor_obs

    def _g1_build_actor_obs(self, motion_time):
        policy_obs = _orig_build_actor_obs(self, motion_time)
        ab = getattr(self, "_g1_ablation", None)
        obs_state: ObsAblationState | None = getattr(self, "_g1_obs_ablation", None)
        if ab is None or obs_state is None or not obs_state.rules:
            return policy_obs
        po = policy_obs.detach().cpu().numpy().reshape(-1)
        single = po[: cp.N_OBS_SINGLE]
        hist_flat = po[cp.N_OBS_SINGLE :]
        applied = self.action_tensor.detach().cpu().numpy().reshape(-1)
        raw_last = getattr(self, "_g1_last_raw_action", None)
        single_ab, diff = obs_state.apply_to_single_obs(
            single, applied_action=applied, raw_action=raw_last
        )
        self._g1_last_obs_ablation_diff = diff
        hist = hist_flat.reshape(self.cfg.history_len, cp.N_OBS_SINGLE)
        for field, mode, _ in obs_state.rules:
            if field == "history" and mode == "repeat_current":
                hist = np.tile(single_ab.reshape(1, -1), (self.cfg.history_len, 1))
        new_po = np.concatenate([single_ab, hist.reshape(-1)]).astype(np.float32)
        return torch.tensor(new_po, dtype=torch.float32, device=self.device).unsqueeze(0)

    cp.MujocoRMGSim._build_actor_obs = _g1_build_actor_obs

    def _g1_run(self, headless=False):
        self.reset()
        if self._jitter_diag is not None:
            self._jitter_diag.begin_episode(self.data)
        self._run_stop_reason = ""
        self._last_motion_time = float(self.motion_time_offset)
        eval_d = self._effective_eval_duration_s()
        if self.cfg.sim_duration is None:
            print(
                f"[sim2sim] playback until eval end "
                f"({self.motion_time_offset:.2f}s → {self._motion_playback_end_time():.2f}s; "
                f"ref={self.ref_duration_s:.2f}s, eval={eval_d:.2f}s, full_ref={self.full_ref})",
                flush=True,
            )
        else:
            end_t = self._motion_playback_end_time()
            print(
                f"[sim2sim] playback sim_duration={self.cfg.sim_duration:.2f}s "
                f"({self.motion_time_offset:.2f}s → {end_t:.2f}s; ref={self.ref_duration_s:.2f}s)",
                flush=True,
            )
        hold_s = float(getattr(self, "viewer_hold_s", 0.0) or 0.0)
        try:
            if headless:
                self._run_sim_loop(headless=True)
            else:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    self._run_sim_loop(headless=False, viewer=viewer)
                    if hold_s > 0.0:
                        print(
                            f"[sim2sim] holding viewer for {hold_s:.0f}s (close window to exit)",
                            flush=True,
                        )
                        t_end = time.time() + hold_s
                        while viewer.is_running() and time.time() < t_end:
                            viewer.sync()
                            time.sleep(0.03)
        finally:
            self._close_inference_log()
            self._write_run_metadata()
            if self._jitter_diag is not None:
                self._jitter_diag.finalize()

    cp.MujocoRMGSim.run = _g1_run

    def _g1_collect_foot_trace(self) -> dict:
        foot_z, _, foot_fn, foot_contact_bool, foot_vel_xy_sq = cp._read_feet_state(
            self.model,
            self.data,
            self._foot_body_ids,
            self._foot_geom_to_idx,
            self._ground_geom_ids,
            contact_force_threshold=float(cp.CP_RMG_V1["feet_contact_force_threshold"]),
        )
        slip = np.sqrt(np.maximum(foot_vel_xy_sq, 0.0))
        prev = getattr(self, "_g1_prev_contact", [False, False])
        switch = [bool(foot_contact_bool[i]) != bool(prev[i]) for i in range(2)]
        self._g1_prev_contact = [bool(foot_contact_bool[0]), bool(foot_contact_bool[1])]
        foot_pos = []
        foot_vel = []
        for bid in self._foot_body_ids:
            foot_pos.append(self.data.xpos[bid].copy())
            foot_vel.append(self.data.cvel[bid, 3:6].copy())
        return {
            "contact_flags": np.asarray(foot_contact_bool, dtype=np.float32),
            "foot_normal_force": np.asarray(foot_fn, dtype=np.float32),
            "foot_tangential_force": np.zeros(2, dtype=np.float32),
            "foot_pos_world": np.stack(foot_pos, axis=0),
            "foot_vel_world": np.stack(foot_vel, axis=0),
            "stance_slip_speed": np.asarray(slip, dtype=np.float32),
            "contact_switch_flags": np.asarray(switch, dtype=np.float32),
        }

    def _g1_step_policy(self, motion_time_clamped_tensor, unclamped_motion_time: float):
        ab: ControlSweepConfig = getattr(self, "_g1_ablation", ControlSweepConfig())
        ctrl_step = int(self.episode_length_buf.item())
        log_pre = None
        if self._inference_log_writer is not None:
            base_quat_xyzw = self._get_base_quat_xyzw()
            base_quat_t = torch.tensor(
                base_quat_xyzw, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            base_ang_vel_local = (
                self._get_base_ang_vel_local(base_quat_t)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            dof_pos_act0, dof_vel_act0 = self._get_dof_state_act()
            log_pre = {
                "timestamp": float(unclamped_motion_time),
                "base_quat_xyzw": base_quat_xyzw,
                "base_ang_vel_local": base_ang_vel_local,
                "joint_pos_policy": self._act_to_policy_vec(dof_pos_act0),
                "joint_vel_policy": self._act_to_policy_vec(dof_vel_act0),
            }

        raw_action = np.zeros((self.cfg.num_actions,), dtype=np.float32)
        scaled_action = raw_action.copy()
        group_ablated_action = raw_action.copy()
        obs_actor_input = None
        replay_mode = ab.replay_target_trace or ab.replay_torque_trace
        trace = getattr(self, "_g1_replay_trace", None)

        if ab.replay_torque_trace and trace is not None:
            n = int(trace["target_after_joint_limit"].shape[0])
            idx = min(ctrl_step, n - 1)
            raw_action = np.asarray(trace.get("raw_action", raw_action)[idx], dtype=np.float32)
            scaled_action = np.asarray(trace.get("scaled_action", raw_action)[idx], dtype=np.float32)
            group_ablated_action = np.asarray(
                trace.get("group_ablated_action", scaled_action)[idx], dtype=np.float32
            )
            action_cmd = np.asarray(trace.get("applied_action", group_ablated_action)[idx], dtype=np.float32)
            tgt_detail = {
                "target_after_joint_limit": np.asarray(
                    trace["target_after_joint_limit"][idx], dtype=np.float32
                )
            }
            target_dof_pos_policy = tgt_detail["target_after_joint_limit"]
            torque_direct = np.asarray(
                trace.get(
                    "torque_after_motor_filter",
                    trace.get("torque_post_clip", trace.get("torque_raw_pre_clip")),
                )[idx],
                dtype=np.float32,
            )
        elif ab.replay_target_trace and trace is not None:
            n = int(trace["target_after_joint_limit"].shape[0])
            idx = min(ctrl_step, n - 1)
            raw_action = np.asarray(trace.get("raw_action", raw_action)[idx], dtype=np.float32)
            scaled_action = np.asarray(trace.get("scaled_action", raw_action)[idx], dtype=np.float32)
            group_ablated_action = np.asarray(
                trace.get("group_ablated_action", scaled_action)[idx], dtype=np.float32
            )
            action_cmd = np.asarray(trace.get("applied_action", group_ablated_action)[idx], dtype=np.float32)
            target_dof_pos_policy = np.asarray(
                trace["target_after_joint_limit"][idx], dtype=np.float32
            )
            tgt_detail = {"target_after_joint_limit": target_dof_pos_policy}
            torque_direct = None
        else:
            policy_motion_time = torch.tensor(
                [float(unclamped_motion_time)],
                dtype=torch.float32,
                device=self.device,
            )
            raw_obs = self._build_actor_obs(policy_motion_time)
            obs_for_onnx = self._clip_raw_obs_for_onnx(raw_obs, self.cfg.clip_observations)
            obs_actor_input = obs_for_onnx.detach().cpu().numpy().reshape(-1).astype(np.float32)
            if bool(getattr(self.cfg, "pure_ref_pd", False)) or self.cfg.zero_policy_action:
                raw_action = np.zeros((self.cfg.num_actions,), dtype=np.float32)
            else:
                raw_action = self._infer_onnx_actions(obs_for_onnx)
            self._g1_last_raw_action = raw_action.copy()
            scaled_action = raw_action * float(ab.policy_action_scale_mult)
            group_scales = parse_ablate_action_groups(ab.ablate_action_groups)
            group_ablated_action = apply_group_action_scale(
                scaled_action, group_scales, self._g1_group_indices
            )
            clip_lim = self.cfg.clip_actions / self.action_scale
            clipped_action = np.clip(group_ablated_action, -clip_lim, clip_lim)
            action_cmd = self._postprocess_policy_action(clipped_action)
            tgt_detail = self._compute_g1_pd_targets_detailed(action_cmd)
            target_dof_pos_policy = tgt_detail["target_after_joint_limit"]
            torque_direct = None

        action_eff = action_cmd
        self.action_tensor[:] = torch.tensor(
            action_cmd, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        target_dof_pos_act = self._policy_to_act_vec(target_dof_pos_policy)
        tau_raw_last = None
        tau_post_last = None
        tau_filt_last = None
        for _ in range(self.cfg.decimation):
            dof_pos_act, dof_vel_act = self._get_dof_state_act()
            if torque_direct is not None:
                tau_raw = torque_direct.copy()
                tau_post, tau_filt = tau_raw, tau_raw
            else:
                tau_raw = (target_dof_pos_act - dof_pos_act) * self.kps_act - dof_vel_act * self.kds_act
                tau_post, tau_filt = self._g1_limit_torques_with_filter(tau_raw, dof_vel_act, ab)
            tau_raw_last = tau_raw
            tau_post_last = tau_post
            tau_filt_last = tau_filt
            self.data.ctrl[:] = tau_filt
            mujoco.mj_step(self.model, self.data)
            if self._jitter_diag is not None:
                dof_after, _ = self._get_dof_state_act()
                dof_pol = self._act_to_policy_vec(dof_after)
                ctx = self._build_diag_track_ctx(
                    motion_time_clamped_tensor,
                    unclamped_motion_time,
                    target_dof_pos_policy,
                    tau_raw,
                    dof_pol,
                    action_eff,
                )
                self._jitter_diag.record_after_step(
                    self.data, action_cmd, target_dof_pos_policy, track_ctx=ctx
                )

        if ab.record_control_trace and getattr(self, "_g1_trace_recorder", None) is not None:
            dof_pos_act, dof_vel_act = self._get_dof_state_act()
            dof_pos_pol = self._act_to_policy_vec(dof_pos_act)
            dof_vel_pol = self._act_to_policy_vec(dof_vel_act)
            foot_info = self._g1_collect_foot_trace()
            base_quat = self._get_base_quat_xyzw()
            _, _, by = _euler_xyz_from_quat_xyzw(np.asarray(base_quat, dtype=np.float64))
            yaw_drift = float(by - self._init_yaw_diag) if self._init_yaw_diag is not None else 0.0
            root_pos, root_rot, _, _, _ = self._query_ref_state(motion_time_clamped_tensor)
            ref_xy = root_pos[0, :2].detach().cpu().numpy()
            base_xy = np.array(self.data.qpos[0:2], dtype=np.float64)
            self._g1_trace_recorder.append(
                time=float(unclamped_motion_time),
                ref_time=float(motion_time_clamped_tensor.reshape(-1)[0].item()),
                obs_raw=None if replay_mode else raw_obs.detach().cpu().numpy().reshape(-1),
                obs_actor_input=obs_actor_input,
                raw_action=raw_action,
                scaled_action=scaled_action,
                group_ablated_action=group_ablated_action,
                applied_action=action_cmd,
                ref_dof_pos=self.ref_dof_pos_res.copy(),
                target_raw=tgt_detail.get("target_raw", target_dof_pos_policy),
                target_after_offset_clip=tgt_detail.get(
                    "target_after_offset_clip", target_dof_pos_policy
                ),
                target_after_joint_limit=target_dof_pos_policy,
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
                dof_pos=dof_pos_pol,
                dof_vel=dof_vel_pol,
                torque_raw_pre_clip=tau_raw_last,
                torque_post_clip=tau_post_last,
                torque_after_motor_filter=tau_filt_last,
                base_pos=self.data.qpos[0:3].copy(),
                base_quat=base_quat,
                base_lin_vel=self.data.qvel[0:3].copy(),
                base_ang_vel=self.data.qvel[3:6].copy(),
                root_xy_err=ref_xy - base_xy,
                yaw_err=0.0,
                yaw_drift=yaw_drift,
                obs_ablation_rms_diff=getattr(self, "_g1_last_obs_ablation_diff", {}),
                **foot_info,
            )

        if log_pre is not None:
            self._append_inference_log(
                log_pre["timestamp"],
                log_pre["base_quat_xyzw"],
                log_pre["base_ang_vel_local"],
                log_pre["joint_pos_policy"],
                log_pre["joint_vel_policy"],
                raw_action,
                action_cmd,
                motion_time_clamped_tensor,
                unclamped_motion_time,
            )

    cp.MujocoRMGSim._step_policy = _g1_step_policy
    cp.MujocoRMGSim._g1_collect_foot_trace = _g1_collect_foot_trace

    def _g1_attach_ablation_runtime(self, ab: ControlSweepConfig) -> None:
        self._g1_ablation = ab
        self._g1_group_indices = build_joint_group_indices(self.cfg.dof_names)
        self._g1_obs_ablation = ObsAblationState(
            parse_obs_ablation(ab.obs_ablation), self.cfg.num_actions
        )
        self._g1_prev_contact = [False, False]
        self._g1_motor_tau_state = np.zeros((self.model.nu,), dtype=np.float32)
        if ab.record_control_trace:
            self._g1_trace_recorder = ControlTraceRecorder(
                ab.record_control_trace, self.cfg.num_actions
            )
        else:
            self._g1_trace_recorder = None
        if ab.replay_target_trace:
            self._g1_replay_trace = load_control_trace(ab.replay_target_trace)
        elif ab.replay_torque_trace:
            self._g1_replay_trace = load_control_trace(ab.replay_torque_trace)
        else:
            self._g1_replay_trace = None
        if ab.ablate_action_groups or ab.policy_action_scale_mult != 1.0:
            group_membership = {
                g: [self.cfg.dof_names[int(i)] for i in idx]
                for g, idx in self._g1_group_indices.items()
                if g not in ("lower", "legs")
            }
            print("[g1-ablation] joint group membership:", group_membership, flush=True)
            print(
                f"[g1-ablation] policy_action_scale_mult={ab.policy_action_scale_mult} "
                f"ablate_action_groups={ab.ablate_action_groups or 'none'}",
                flush=True,
            )

    cp.MujocoRMGSim._g1_attach_ablation_runtime = _g1_attach_ablation_runtime

    def _g1_reset_action_delay_steps(self) -> None:
        if self.cfg.action_delay_steps is not None:
            delay = int(self.cfg.action_delay_steps)
        elif self.cfg.action_delay:
            lo, hi = cp.CP_RMG_V1.get("ctrl_delay_step_bounds", (1, 6))
            delay = int(np.random.randint(int(lo), int(hi) + 1))
        else:
            delay = 0
        max_delay = max(0, min(delay, self.cfg.action_buf_len - 1))
        self._action_delay_steps = max_delay

    cp.MujocoRMGSim._reset_action_delay_steps = _g1_reset_action_delay_steps

    def _configure_model_with_g1_armature(self):
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            self.model.geom_friction[floor_id][0] = self.cfg.friction_coeffs

        armature_act_order = self._policy_to_act_vec(_G1_JOINT_ARMATURE)
        for act_i, dof_adr in enumerate(self._act_qveladr):
            self.model.dof_armature[dof_adr] = float(armature_act_order[act_i])

    cp.MujocoRMGSim._configure_model = _configure_model_with_g1_armature

    _orig_reset = cp.MujocoRMGSim.reset

    def _g1_reset(self):
        _orig_reset(self)
        if hasattr(self, "_g1_ablation"):
            self._g1_motor_tau_state = np.zeros((self.model.nu,), dtype=np.float32)
            self._g1_prev_contact = [False, False]

        if bool(getattr(self.cfg, "g1_balance_safe_enable", False)):
            # Preserve the legacy v4 balance-safe sole-clearance reset.
            clearance = float(cp.CP_RMG_V1.get("initial_sole_clearance_m", 0.005))
            ankle_2g = float(cp.CP_RMG_V1["ankle_2_ground"])
            foot_z, _, _, _, _ = cp._read_feet_state(
                self.model,
                self.data,
                self._foot_body_ids,
                self._foot_geom_to_idx,
                self._ground_geom_ids,
                contact_force_threshold=float(cp.CP_RMG_V1["feet_contact_force_threshold"]),
            )
            sole_z = foot_z - ankle_2g
            if np.all(np.isfinite(sole_z)):
                self.data.qpos[2] += clearance - float(np.min(sole_z))
        mujoco.mj_forward(self.model, self.data)

    cp.MujocoRMGSim.reset = _g1_reset


def _load_and_validate_run_contract(
    path: str | None,
    *,
    args: argparse.Namespace,
) -> dict | None:
    if not path:
        return None
    contract_path = Path(path).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"run contract not found: {contract_path}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run contract: {contract_path}") from exc
    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(
            f"unsupported or missing evaluation contract schema in {contract_path}"
        )
    try:
        validate_canonical_abi(contract.get("canonical_abi"))
    except ValueError as exc:
        raise ValueError(f"evaluation contract canonical ABI mismatch: {exc}") from exc
    if not isinstance(contract.get("run_id"), str) or not contract["run_id"]:
        raise ValueError("evaluation contract must contain a non-empty run_id")
    if not isinstance(contract.get("seed"), int):
        raise ValueError("evaluation contract must contain an integer seed")
    if int(args.seed) != int(contract["seed"]):
        raise ValueError(
            f"seed mismatch: runner={args.seed} contract={contract['seed']}"
        )
    control = contract.get("control")
    if not isinstance(control, dict):
        raise ValueError("evaluation contract missing control object")
    if args.v4:
        raise ValueError("contract-bound MAGE80250 runs cannot use --v4")
    if args.no_action_delay or args.no_action_lowpass:
        raise ValueError("contract-bound runs cannot disable delay or low-pass")
    if control.get("action_delay_enabled") is not True:
        raise ValueError("evaluation contract must keep action delay enabled")
    if control.get("action_filter_enabled") is not True:
        raise ValueError("evaluation contract must keep action low-pass enabled")
    if args.action_delay_steps is None or args.action_filter_alpha is None:
        raise ValueError(
            "contract-bound runs require explicit --action-delay-steps and "
            "--action-filter-alpha"
        )
    if int(args.action_delay_steps) != int(control.get("delay_steps", -1)):
        raise ValueError("action delay does not match evaluation contract")
    if abs(float(args.action_filter_alpha) - float(control.get("action_filter_alpha", -1.0))) > 1.0e-6:
        raise ValueError("action filter alpha does not match evaluation contract")
    if abs(float(args.motor_tau_s) - float(control.get("motor_tau_s", -1.0))) > 1.0e-9:
        raise ValueError("motor tau does not match evaluation contract")
    if tuple(control.get("candidate_delays", ())) != (0, 1, 2):
        raise ValueError("evaluation contract candidate delays differ from canonical ABI")
    if tuple(float(value) for value in control.get("candidate_alphas", ())) != (0.4, 0.5, 0.6):
        raise ValueError("evaluation contract candidate alphas differ from canonical ABI")

    inputs = contract.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("evaluation contract missing inputs object")
    actual_paths = {
        "policy_onnx": Path(args.onnx),
        "policy_onnx_sidecar": Path(f"{args.onnx}.data"),
        "policy_pt": Path(str((inputs.get("policy_pt") or {}).get("path", ""))),
        "export_manifest": Path(str((inputs.get("export_manifest") or {}).get("path", ""))),
        "exporter": Path(str((inputs.get("exporter") or {}).get("path", ""))),
        "motion_csv": Path(args.motion_csv),
        "motion_pool_manifest": Path(
            str((inputs.get("motion_pool_manifest") or {}).get("path", ""))
        ),
        "xml": Path(args.xml),
        "runner": Path(__file__),
        "base_runner": Path(cp.__file__),
        "jitter_helper": Path(jitter_diag_module.__file__),
        "contract_helper": _REPO_ROOT / "tw_cp_mujoco_119" / "scripts" / "evaluation_contract.py",
    }
    expected_topology = contract.get("xml_topology")
    if not isinstance(expected_topology, dict):
        raise ValueError("evaluation contract missing xml_topology object")
    actual_topology = inspect_xml_topology(args.xml)
    for key in ("wrist_roll_mode", "fixed_wrist_roll_joints", "joint_count_in_policy"):
        if actual_topology.get(key) != expected_topology.get(key):
            raise ValueError(
                f"XML topology differs from evaluation contract at {key}: "
                f"actual={actual_topology.get(key)!r} expected={expected_topology.get(key)!r}"
            )
    for key, actual_path in actual_paths.items():
        expected = inputs.get(key)
        if not isinstance(expected, dict):
            raise ValueError(f"evaluation contract missing input identity: {key}")
        if key == "policy_onnx_sidecar" and expected.get("exists") is False:
            if actual_path.exists():
                raise ValueError("contract expects no ONNX external-data sidecar, but one exists")
            if str(actual_path.resolve()) != str(expected.get("path")):
                raise ValueError("contract sidecar path does not match the policy path")
            continue
        actual = file_identity(actual_path, role=str(expected.get("role", key)))
        for field in ("path", "size_bytes", "sha256"):
            if actual[field] != expected.get(field):
                raise ValueError(
                    f"evaluation contract identity mismatch for {key}.{field}: "
                    f"actual={actual[field]!r} expected={expected.get(field)!r}"
                )
    export_identity = inputs.get("export_manifest")
    if not isinstance(export_identity, dict):
        raise ValueError("evaluation contract missing export manifest identity")
    export_path = Path(str(export_identity.get("path", ""))).resolve()
    try:
        export_payload = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid policy export manifest: {export_path}") from exc
    if not isinstance(export_payload, dict) or export_payload.get("schema") != "g1-mage-policy-export/v1":
        raise ValueError("unsupported policy export manifest schema")
    if int(export_payload.get("return_code", -1)) != 0:
        raise ValueError("policy export manifest is not a successful export")
    exported = export_payload.get("output_onnx")
    source_pt = export_payload.get("source_pt")
    if not isinstance(exported, dict) or not isinstance(source_pt, dict):
        raise ValueError("policy export manifest missing output_onnx/source_pt")
    if Path(str(exported.get("path", ""))).resolve() != Path(args.onnx).resolve():
        raise ValueError("runner ONNX is not the ONNX bound by export manifest")
    if exported.get("shapes") != {"input": ["batch", 1166], "output": ["batch", 21]}:
        raise ValueError("runner ONNX shapes do not match the canonical ABI")
    if exported.get("normalizer") != "none":
        raise ValueError("contract-bound MAGE80250 runner requires a raw ONNX export")
    expected_pt = inputs.get("policy_pt")
    if not isinstance(expected_pt, dict):
        raise ValueError("policy export PT identity is missing from evaluation contract")
    for field in ("path", "size_bytes", "sha256"):
        if source_pt.get(field) != expected_pt.get(field):
            raise ValueError(f"policy export PT identity mismatch at {field}")
    expected_exporter = inputs.get("exporter")
    payload_exporter = export_payload.get("exporter")
    if not isinstance(expected_exporter, dict) or not isinstance(payload_exporter, dict):
        raise ValueError("policy export exporter identity is missing")
    for field in ("path", "size_bytes", "sha256"):
        if payload_exporter.get(field) != expected_exporter.get(field):
            raise ValueError(f"policy export exporter identity mismatch at {field}")
    if source_pt.get("sha256") != expected_pt.get("sha256"):
        raise ValueError("policy export PT identity does not match evaluation contract")
    actual_exported = file_identity(args.onnx, role="policy_onnx")
    for field in ("path", "size_bytes", "sha256"):
        if actual_exported[field] != exported.get(field):
            raise ValueError(f"policy export ONNX identity mismatch at {field}")
    selected = (contract.get("motion_selection") or {}).get("selected")
    if not isinstance(selected, dict) or selected.get("path") != str(actual_paths["motion_csv"].resolve()):
        raise ValueError("motion CSV is not the explicitly selected contract motion")
    horizon = contract.get("horizon")
    if not isinstance(horizon, dict):
        raise ValueError("evaluation contract missing horizon object")
    expected_hold = float(horizon.get("post_motion_hold_s", -1.0))
    if abs(expected_hold - POST_MOTION_HOLD_S) > 1.0e-6:
        raise ValueError(
            f"contract terminal hold must be {POST_MOTION_HOLD_S}s, got {expected_hold}"
        )
    if not args.hold_last_ref or abs(float(args.post_motion_hold_s) - expected_hold) > 1.0e-6:
        raise ValueError("contract-bound runs require the explicit terminal hold")
    if horizon.get("terminal_velocity_decay") != "cosine":
        raise ValueError("contract-bound runs require cosine terminal velocity decay")
    expected_requested = float(horizon.get("runner_sim_duration_s", -1.0))
    expected_effective = float(horizon.get("effective_horizon_s", -1.0))
    if args.sim_duration is None or abs(float(args.sim_duration) - expected_requested) > 1.0e-6:
        raise ValueError("contract-bound sim duration does not match planned horizon")
    if (
        args.max_episode_length_s is None
        or abs(float(args.max_episode_length_s) - expected_effective) > 1.0e-6
    ):
        raise ValueError("contract-bound max episode length does not match effective horizon")
    outputs = contract.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("evaluation contract missing outputs object")
    output_bindings = {
        "run_contract": path,
        "log_csv": args.inference_log_csv,
        "diagnostics_dir": args.diagnostics_dir,
        "run_metadata_json": args.run_metadata_json,
    }
    for key, actual_value in output_bindings.items():
        expected_value = outputs.get(key)
        if not isinstance(expected_value, str) or not isinstance(actual_value, str):
            raise ValueError(f"evaluation contract output binding missing: {key}")
        if Path(actual_value).resolve() != Path(expected_value).resolve():
            raise ValueError(f"evaluation contract output path mismatch: {key}")
    contract["_path"] = str(contract_path)
    contract["_sha256"] = sha256_file(contract_path)
    return contract


def _assert_xml_has_floor(xml_path: str) -> None:
    floor_id = mujoco.mj_name2id(
        mujoco.MjModel.from_xml_path(xml_path), mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if floor_id < 0:
        raise RuntimeError(
            f"MJCF 缺少地面 geom 'floor'（plane），机器人会无限下落: {xml_path}\n"
            "请使用含 floor 的 g1_21dof.xml，或运行 tools/generate_g1_21dof_xml.py 重新生成。"
        )


def _validate_g1_canonical_config(cfg: cp.RMGConfig) -> None:
    """Check the effective runner config against the dependency-free ABI."""
    validate_canonical_abi(CANONICAL_ABI)
    labels = build_g1_student327_single_obs_labels()
    observed_scales = (
        [1.0] * 36
        + [float(cfg.obs_ang_vel_scale)] * 3
        + [1.0] * 4
        + [float(cfg.obs_dof_pos_scale)] * 21
        + [float(cfg.obs_dof_vel_scale)] * 21
        + [1.0] * 21
    )
    if tuple(labels) != tuple(CANONICAL_OBSERVATION_FIELDS):
        raise ValueError("G1 student327 observation labels differ from canonical ABI")
    if tuple(float(value) for value in observed_scales) != tuple(CANONICAL_OBSERVATION_SCALES):
        raise ValueError("G1 student327 observation scales differ from canonical ABI")
    cp.validate_canonical_runner_observation_layout(
        labels,
        observed_scales,
        single_dim=G1_STUDENT327_N_OBS_SINGLE,
        history_len=cfg.history_len,
        policy_dim=G1_STUDENT327_N_POLICY_OBS,
        ankle_velocity_indices=cfg.ankle_idx,
    )
    if tuple(cfg.dof_names) != tuple(CANONICAL_POLICY_JOINT_ORDER):
        raise ValueError("runner policy joint order differs from canonical ABI")
    if int(cfg.num_actions) != int(CANONICAL_ABI["policy_io"]["output_dim"]):
        raise ValueError("runner action count differs from canonical ABI")
    if int(cfg.history_len) != int(CANONICAL_ABI["policy_io"]["history_length"]):
        raise ValueError("runner history length differs from canonical ABI")
    if abs(float(cfg.simulation_dt) * int(cfg.decimation) - 1.0 / 50.0) > 1.0e-9:
        raise ValueError("runner control rate differs from canonical 50Hz ABI")
    if abs(float(cfg.root_reset_z_offset) - 0.05) > 1.0e-9:
        raise ValueError("runner root reset z offset differs from canonical ABI")
    if abs(float(cfg.clip_actions) - 0.6) > 1.0e-9:
        raise ValueError("runner residual clip differs from canonical ABI")
    if tuple(float(x) for x in cfg.action_scale) != tuple(CANONICAL_ACTION_SCALE):
        raise ValueError("runner action scale differs from canonical ABI")
    if not np.allclose(np.asarray(cfg.kps, dtype=np.float64), CANONICAL_KP, atol=1.0e-6, rtol=0.0):
        raise ValueError("runner Kp differs from canonical ABI")
    if not np.allclose(np.asarray(cfg.kds, dtype=np.float64), CANONICAL_KD, atol=1.0e-6, rtol=0.0):
        raise ValueError("runner Kd differs from canonical ABI")
    if not bool(cfg.action_delay) or int(cfg.action_delay_steps or -1) != BASELINE_DELAY_STEPS:
        raise ValueError("runner baseline action delay differs from canonical ABI")
    if not bool(cfg.action_lowpass_filter) or abs(
        float(cfg.action_filter_alpha) - BASELINE_ACTION_FILTER_ALPHA
    ) > 1.0e-9:
        raise ValueError("runner baseline low-pass differs from canonical ABI")


def make_g1_rmg_config(**overrides):
    overrides.setdefault("urdf_path", str(_G1_URDF))
    extra: dict = {}
    for key in _G1_CFG_EXTRA_KEYS:
        if key in overrides:
            extra[key] = overrides.pop(key)
        elif key in cp.CP_RMG_V1:
            extra[key] = cp.CP_RMG_V1[key]
    cfg = cp.make_rmg_config(**overrides)
    for key, val in extra.items():
        setattr(cfg, key, val)
    if not hasattr(cfg, "pure_ref_pd"):
        setattr(cfg, "pure_ref_pd", False)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo ONNX inference for G1 RMG (CSV direct input)")
    parser.add_argument("--xml", default=str(_DEFAULT_XML))
    parser.add_argument("--onnx", default=str(_DEFAULT_ONNX))
    parser.add_argument("--skip-onnx-normalizer-check", action="store_true")
    parser.add_argument("--motion-csv", default=str(_DEFAULT_MOTION_CSV))
    parser.add_argument("--csv-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=None,
        help="评测 horizon；未指定时为 motion end + --post-motion-hold-s",
    )
    parser.add_argument(
        "--max-episode-length-s",
        type=float,
        default=None,
        help="可选全局 episode 上限；小于计划 horizon 时记录 horizon_truncated",
    )
    parser.add_argument(
        "--post-motion-hold-s",
        type=float,
        default=POST_MOTION_HOLD_S,
        help="motion end 后末帧保持/余弦速度衰减时长（秒）",
    )
    parser.add_argument(
        "--hold-last-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="参考结束后保持最后一帧并继续推理到 episode 上限",
    )
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument(
        "--ref-turn-intent-horizon-s",
        type=float,
        default=cp.REF_TURN_INTENT_HORIZON_S,
    )
    parser.add_argument("--ankle-pitch-bias-left", type=float, default=0.0)
    parser.add_argument("--ankle-pitch-bias-right", type=float, default=0.0)
    parser.add_argument("--v1-clean", action="store_true")
    parser.add_argument(
        "--v4",
        action="store_true",
        help="对齐 g1_rmg_v4_s2s_repair：clip_actions=0.5、无 delay/低通、balance-safe 目标裁剪",
    )
    parser.add_argument(
        "--action-filter-alpha",
        type=float,
        default=BASELINE_ACTION_FILTER_ALPHA,
    )
    parser.add_argument(
        "--no-action-lowpass",
        action="store_true",
        default=None,
        help="关闭 action 低通；--v4 时默认关闭",
    )
    parser.add_argument(
        "--no-action-delay",
        action="store_true",
        default=None,
        help="关闭 action delay；--v4 时默认关闭",
    )
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=BASELINE_DELAY_STEPS,
    )
    parser.add_argument(
        "--motor-tau-s",
        type=float,
        default=BASELINE_MOTOR_TAU_S,
        help="motor torque 一阶滤波时间常数；baseline contract 固定为 0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-contract-json",
        type=str,
        default=None,
        help="fail-closed evaluation contract JSON；由 batch evaluator 显式提供",
    )
    parser.add_argument(
        "--run-metadata-json",
        type=str,
        default=None,
        help="写入本次 rollout 的 effective contract/termination metadata",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--viewer-hold-s",
        type=float,
        default=30.0,
        help="仿真结束后保持 MuJoCo 窗口的秒数（关掉窗口可提前退出）",
    )
    parser.add_argument(
        "--pure-ref-pd",
        action="store_true",
        help="跳过 ONNX，target=ref_dof_pos 经 balance-safe 裁剪后 PD（非动态稳定性 gate）",
    )
    parser.add_argument(
        "--zero-policy-action",
        action="store_true",
        help="策略输出置零；仅用于短窗口 ABI sanity check，不用于全身动态稳定性评价",
    )
    parser.add_argument(
        "--policy-action-scale-mult",
        type=float,
        default=1.0,
        help="policy-in-loop：将 actor residual 按比例缩放后再 clip/delay/PD（默认 1.0）",
    )
    parser.add_argument(
        "--ablate-action-groups",
        type=str,
        default="",
        help="关节组 residual 缩放，如 legs=1.0,arms=0.0 或 ankle=0.5,hip_roll=0.5",
    )
    parser.add_argument("--kp-scale", type=float, default=1.0, help="全局 PD kp 缩放（默认 1.0）")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="全局 PD kd 缩放（默认 1.0）")
    parser.add_argument(
        "--torque-limit-scale",
        type=float,
        default=1.0,
        help="力矩限幅缩放（默认 1.0）",
    )
    parser.add_argument("--ankle-kp-scale", type=float, default=1.0, help="踝 kp 缩放")
    parser.add_argument("--ankle-kd-scale", type=float, default=1.0, help="踝 kd 缩放")
    parser.add_argument("--hip-roll-kp-scale", type=float, default=1.0, help="髋 roll kp 缩放")
    parser.add_argument("--hip-roll-kd-scale", type=float, default=1.0, help="髋 roll kd 缩放")
    parser.add_argument(
        "--obs-ablation",
        type=str,
        default="",
        help="观测消融，如 ankle_vel=raw,base_ang_vel=lowpass:0.3,previous_action=zero",
    )
    parser.add_argument(
        "--record-control-trace",
        type=str,
        default=None,
        help="policy-in-loop 运行时记录 control trace 到 npz",
    )
    parser.add_argument(
        "--replay-target-trace",
        type=str,
        default=None,
        help="target shadow replay：从 trace 读取 target_after_joint_limit，不跑 actor",
    )
    parser.add_argument(
        "--replay-torque-trace",
        type=str,
        default=None,
        help="torque shadow replay：从 trace 读取 torque，不跑 actor/PD",
    )
    parser.add_argument("--zero-ankle-roll-action", action="store_true")
    parser.add_argument("--zero-hip-roll-and-ankle-roll-action", action="store_true")
    parser.add_argument("--diagnose-jitter-slip", action="store_true")
    parser.add_argument("--diagnose-sim2sim", action="store_true")
    parser.add_argument("--diagnostics-dir", type=str, default=None)
    parser.add_argument("--jitter-cutoff-hz", type=float, default=25.0)
    parser.add_argument("--diagnostics-note", type=str, default="")
    parser.add_argument("--inference-log-csv", default=None)
    return parser.parse_args()


def _ablation_from_args(args: argparse.Namespace) -> ControlSweepConfig:
    return ControlSweepConfig(
        policy_action_scale_mult=float(args.policy_action_scale_mult),
        ablate_action_groups=str(args.ablate_action_groups or ""),
        obs_ablation=str(args.obs_ablation or ""),
        kp_scale=float(args.kp_scale),
        kd_scale=float(args.kd_scale),
        ankle_kp_scale=float(args.ankle_kp_scale),
        ankle_kd_scale=float(args.ankle_kd_scale),
        hip_roll_kp_scale=float(args.hip_roll_kp_scale),
        hip_roll_kd_scale=float(args.hip_roll_kd_scale),
        torque_limit_scale=float(args.torque_limit_scale),
        motor_tau_s=float(args.motor_tau_s),
        record_control_trace=args.record_control_trace,
        replay_target_trace=args.replay_target_trace,
        replay_torque_trace=args.replay_torque_trace,
    )


def _write_policy_in_loop_summary(
    diag_dir: str,
    *,
    jitter_summary: dict,
    ablation: ControlSweepConfig,
    motion_csv: str,
    onnx_path: str,
    termination_reason: str,
    sim_duration: float,
    use_v4: bool,
    no_action_delay: bool,
    no_action_lowpass: bool,
    action_filter_alpha: float | None,
    run_contract: dict | None,
    actual_duration_s: float,
    effective_horizon_s: float,
    first_failure_time_s: float | None,
    first_failure_reason: str,
    seed: int,
    limit_contract: dict | None,
) -> None:
    summary_path = os.path.join(diag_dir, "summary.json")
    base: dict = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as fp:
                base = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid pre-existing diagnostics summary: {summary_path}") from exc
    if not isinstance(base, dict):
        raise ValueError(f"diagnostics summary must be an object: {summary_path}")
    trace_arrays = None
    if ablation.record_control_trace and os.path.isfile(ablation.record_control_trace):
        trace_arrays = load_control_trace(ablation.record_control_trace)
    ckpt = None
    m = re.search(r"_(\d+)_actor", os.path.basename(onnx_path))
    if m:
        ckpt = m.group(1)
    meta = {
        **ablation.to_metadata(),
        "sim_duration": sim_duration,
        "no_action_delay": no_action_delay,
        "no_action_lowpass": no_action_lowpass,
        "action_filter_alpha": action_filter_alpha,
        "effective_horizon_s": effective_horizon_s,
        "seed": int(seed),
        "v4": use_v4,
        "g1_balance_safe_enable": True if use_v4 else bool(
            getattr(cp.CP_RMG_V1, "g1_balance_safe_enable", False)
        ),
    }
    enhanced = build_enhanced_summary(
        jitter_summary=base,
        ablation_meta=meta,
        motion_name=Path(motion_csv).stem,
        checkpoint_step=ckpt,
        termination_reason=termination_reason,
        control_dt=float(cp.CP_RMG_V1["simulation_dt"]) * int(cp.CP_RMG_V1["decimation"]),
        group_indices=build_joint_group_indices(G1_POLICY_DOF_NAMES),
        trace_arrays=trace_arrays,
    )
    failed = termination_reason not in getattr(
        cp, "_NON_FAIL_TERMINATION_REASONS", frozenset({""})
    )
    enhanced.update(
        {
            "failed": bool(failed),
            "first_failure_time_s": first_failure_time_s,
            "first_failure_reason": first_failure_reason or None,
            "effective_horizon_s": float(effective_horizon_s),
            "actual_duration": float(actual_duration_s),
            "survival_denominator_s": float(effective_horizon_s),
            "survival_time_s": float(
                first_failure_time_s if failed and first_failure_time_s is not None else actual_duration_s
            ),
            "truncated": termination_reason == "horizon_truncated",
            "limit_contract": limit_contract,
            "evaluation_contract": (
                {
                    "schema": run_contract.get("schema"),
                    "run_id": run_contract.get("run_id"),
                    "evaluation_id": run_contract.get("evaluation_id"),
                    "path": run_contract.get("_path"),
                    "sha256": run_contract.get("_sha256"),
                }
                if run_contract is not None
                else None
            ),
        }
    )
    base["duration_s"] = float(actual_duration_s)
    base["termination_reason"] = termination_reason
    base["failed"] = bool(failed)
    base["first_failure_time_s"] = first_failure_time_s
    base["first_failure_reason"] = first_failure_reason or None
    base["effective_horizon_s"] = float(effective_horizon_s)
    base["canonical_abi"] = canonical_abi()
    base["evaluation_contract"] = enhanced["evaluation_contract"]
    base["summary_schema"] = "g1-mage-sim2sim-summary/v2"
    base["policy_in_loop_summary"] = enhanced
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(base, fp, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.csv_fps <= 0.0 or not np.isfinite(args.csv_fps):
        raise ValueError("--csv-fps must be finite and > 0")
    if args.start_time < 0.0 or not np.isfinite(args.start_time):
        raise ValueError("--start-time must be finite and >= 0")
    if args.sim_duration is not None and (
        args.sim_duration <= 0.0 or not np.isfinite(args.sim_duration)
    ):
        raise ValueError("--sim-duration must be finite and > 0")
    if args.max_episode_length_s is not None and (
        args.max_episode_length_s <= 0.0 or not np.isfinite(args.max_episode_length_s)
    ):
        raise ValueError("--max-episode-length-s must be finite and > 0")
    if args.post_motion_hold_s < 0.0 or not np.isfinite(args.post_motion_hold_s):
        raise ValueError("--post-motion-hold-s must be finite and >= 0")
    if args.action_delay_steps is not None and not 0 <= args.action_delay_steps <= 2:
        raise ValueError("--action-delay-steps must be in the locked range 0..2")
    if args.action_filter_alpha is not None and not 0.0 < args.action_filter_alpha <= 1.0:
        raise ValueError("--action-filter-alpha must be finite and in (0, 1]")
    if args.motor_tau_s < 0.0 or not np.isfinite(args.motor_tau_s):
        raise ValueError("--motor-tau-s must be finite and >= 0")
    np.random.seed(int(args.seed))
    run_contract = _load_and_validate_run_contract(args.run_contract_json, args=args)
    ablation = _ablation_from_args(args)
    _apply_g1_patches(use_v4=bool(args.v4), ablation=ablation)
    no_action_lowpass = args.no_action_lowpass
    no_action_delay = args.no_action_delay
    if args.v4:
        if no_action_lowpass is None:
            no_action_lowpass = True
        if no_action_delay is None:
            no_action_delay = True
    if no_action_lowpass is None:
        no_action_lowpass = False
    if no_action_delay is None:
        no_action_delay = False

    cfg_overrides: dict = {
        "ref_turn_intent_horizon_s": float(args.ref_turn_intent_horizon_s),
        "ankle_pitch_bias_left": float(args.ankle_pitch_bias_left),
        "ankle_pitch_bias_right": float(args.ankle_pitch_bias_right),
        "zero_policy_action": bool(args.zero_policy_action),
        "zero_ankle_roll_action": bool(args.zero_ankle_roll_action),
        "zero_hip_roll_and_ankle_roll_action": bool(args.zero_hip_roll_and_ankle_roll_action),
        "pure_ref_pd": bool(args.pure_ref_pd),
        "inference_log_csv": args.inference_log_csv,
    }
    if no_action_lowpass:
        cfg_overrides["action_lowpass_filter"] = False
        cfg_overrides["action_filter_alpha"] = 1.0
    elif args.action_filter_alpha is not None:
        cfg_overrides["action_filter_alpha"] = float(args.action_filter_alpha)
    if no_action_delay:
        cfg_overrides["action_delay"] = False
    elif args.action_delay_steps is not None:
        cfg_overrides["action_delay"] = True
        cfg_overrides["action_delay_steps"] = int(args.action_delay_steps)
    cfg = make_g1_rmg_config(
        device=args.device,
        sim_duration=None,
        v1_clean=bool(args.v1_clean),
        **cfg_overrides,
    )
    _assert_xml_has_floor(args.xml)
    sim = cp.MujocoRMGSim(
        args.xml,
        args.onnx,
        args.motion_csv,
        args.csv_fps,
        cfg,
        skip_onnx_normalizer_check=bool(args.skip_onnx_normalizer_check)
        or ablation.is_replay_mode,
    )
    sim.viewer_hold_s = float(args.viewer_hold_s)
    if run_contract is not None:
        _validate_g1_canonical_config(cfg)
        sim._canonical_abi = canonical_abi()
    sim._g1_attach_ablation_runtime(ablation)
    if hasattr(sim, "_g1_apply_pd_scales"):
        sim._g1_apply_pd_scales(ablation)
    sim.motion_time_offset = max(0.0, min(args.start_time, sim.motion_len - 0.05))
    sim.ref_end_hold_enabled = bool(args.hold_last_ref)
    sim.post_motion_hold_s = float(args.post_motion_hold_s)
    sim.ref_end_hold_s = float(args.post_motion_hold_s) if args.hold_last_ref else 0.0
    requested_horizon_s = (
        float(args.sim_duration)
        if args.sim_duration is not None
        else float(sim.ref_duration_s) + float(sim.ref_end_hold_s)
    )
    effective_horizon_s = requested_horizon_s
    if args.max_episode_length_s is not None:
        effective_horizon_s = min(effective_horizon_s, float(args.max_episode_length_s))
    if effective_horizon_s <= 0.0:
        raise ValueError("effective evaluation horizon must be positive")
    sim.requested_horizon_s = requested_horizon_s
    sim.effective_horizon_s = effective_horizon_s
    sim.cfg.sim_duration = effective_horizon_s
    sim.max_episode_length_s = effective_horizon_s
    sim.zero_ref_vel_after_end = bool(args.hold_last_ref)
    if run_contract is not None:
        expected_horizon = float(
            (run_contract.get("horizon") or {}).get("effective_horizon_s", -1.0)
        )
        if abs(expected_horizon - effective_horizon_s) > 1.0e-6:
            raise ValueError(
                f"effective horizon mismatch: runner={effective_horizon_s} "
                f"contract={expected_horizon}"
            )
        selected = (run_contract.get("motion_selection") or {}).get("selected") or {}
        expected_motion_duration = float(selected.get("duration_s", -1.0))
        if abs(expected_motion_duration - float(sim.motion_len)) > 1.0e-5:
            raise ValueError(
                f"motion duration mismatch: runner={sim.motion_len} "
                f"contract={expected_motion_duration}"
            )
        sim._evaluation_contract_binding = {
            "schema": run_contract["schema"],
            "run_id": run_contract["run_id"],
            "evaluation_id": run_contract.get("evaluation_id"),
            "path": run_contract["_path"],
            "sha256": run_contract["_sha256"],
        }
    sim.run_metadata_json = (
        os.path.abspath(args.run_metadata_json) if args.run_metadata_json else None
    )
    if args.diagnose_jitter_slip or args.diagnose_sim2sim:
        parent = args.diagnostics_dir or os.path.join(str(_PRJ_DIR), "diagnostics")
        if run_contract is not None and not args.diagnostics_dir:
            raise ValueError("contract-bound runs require an explicit diagnostics directory")
        out_dir = os.path.abspath(args.diagnostics_dir) if args.diagnostics_dir else os.path.join(
            parent, time.strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(out_dir, exist_ok=False)
        meta = {
            "argv": sys.argv,
            "xml": os.path.abspath(args.xml),
            "onnx": os.path.abspath(args.onnx),
            "motion_csv": os.path.abspath(args.motion_csv),
            "csv_fps": args.csv_fps,
            "note": args.diagnostics_note or None,
            "v4": bool(args.v4),
            "pure_ref_pd": bool(getattr(cfg, "pure_ref_pd", False)),
            "zero_policy_action": cfg.zero_policy_action,
            "zero_ankle_roll_action": cfg.zero_ankle_roll_action,
            "zero_hip_roll_and_ankle_roll_action": cfg.zero_hip_roll_and_ankle_roll_action,
            "g1_balance_safe_enable": bool(getattr(cfg, "g1_balance_safe_enable", False)),
            "ankle_roll_action_indices": cp.ANKLE_ROLL_ACTION_INDICES,
            "hip_roll_action_indices": cp.HIP_ROLL_ACTION_INDICES,
            "sim_duration": cfg.sim_duration,
            "requested_horizon_s": requested_horizon_s,
            "effective_horizon_s": effective_horizon_s,
            "post_motion_hold_s": sim.post_motion_hold_s,
            "seed": int(args.seed),
            "evaluation_contract": sim._evaluation_contract_binding
            if run_contract is not None
            else None,
            "canonical_abi": getattr(sim, "_canonical_abi", None),
            "limit_contract": getattr(sim, "_g1_limit_contract_metadata", None),
            "robot": "g1_21dof",
            **ablation.to_metadata(),
        }
        jcfg = JitterSlipDiagnosticsConfig(
            physics_dt=cfg.simulation_dt,
            output_dir=out_dir,
            jitter_cutoff_hz=float(args.jitter_cutoff_hz),
            foot_body_names=tuple(cp._FOOT_BODY_NAMES),
            ankle_policy_indices=tuple(int(x) for x in cfg.ankle_idx),
            run_metadata=meta,
        )
        sim._jitter_diag = JitterSlipDiagnostics(sim.model, jcfg)
    if args.inference_log_csv:
        print(f"[sim2sim] inference CSV: {os.path.abspath(args.inference_log_csv)}", flush=True)
    sim.run(headless=args.headless)
    if getattr(sim, "_g1_trace_recorder", None) is not None:
        trace_path = sim._g1_trace_recorder.save()
        print(f"[g1-ablation] control trace -> {trace_path}", flush=True)
    if args.diagnose_jitter_slip or args.diagnose_sim2sim:
        diag_dir = sim._jitter_diag.cfg.output_dir if sim._jitter_diag is not None else None
        if diag_dir:
            jitter_summary_path = os.path.join(diag_dir, "summary.json")
            jitter_summary = {}
            if os.path.isfile(jitter_summary_path):
                with open(jitter_summary_path, encoding="utf-8") as fp:
                    jitter_summary = json.load(fp)
            _write_policy_in_loop_summary(
                diag_dir,
                jitter_summary=jitter_summary,
                ablation=ablation,
                motion_csv=args.motion_csv,
                onnx_path=args.onnx,
                termination_reason=getattr(sim, "_run_stop_reason", ""),
                sim_duration=float(effective_horizon_s),
                use_v4=bool(args.v4),
                no_action_delay=bool(no_action_delay),
                no_action_lowpass=bool(no_action_lowpass),
                action_filter_alpha=args.action_filter_alpha,
                run_contract=run_contract,
                actual_duration_s=max(
                    0.0,
                    float(sim._last_motion_time) - float(sim.motion_time_offset),
                ),
                effective_horizon_s=float(effective_horizon_s),
                first_failure_time_s=getattr(sim, "_first_failure_time_s", None),
                first_failure_reason=getattr(sim, "_first_failure_reason", ""),
                seed=int(args.seed),
                limit_contract=getattr(sim, "_g1_limit_contract_metadata", None),
            )
    if args.inference_log_csv and os.path.isfile(args.inference_log_csv):
        with open(args.inference_log_csv, encoding="utf-8") as fp:
            nrows = max(sum(1 for _ in fp) - 1, 0)
        print(
            f"[sim2sim] inference CSV wrote {nrows} rows -> {os.path.abspath(args.inference_log_csv)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
