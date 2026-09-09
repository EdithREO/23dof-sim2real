"""
MuJoCo sim2sim：ONNX Actor + CSV 参考轨迹。

部署契约（与 C306MimicNative/C306Mimic_V1 + C306RMGRealRepairCfg + save_cp_stu ObsNormActor 一致）：
  1. obs：125 维 student_obs（mimic 30 + proprio 95）× history_len(10) → 1375 维 raw 向量；
     布局 [current_125, hist_t-1, …, hist_t-10]，与 env.obs_buf 相同。
     mimic 航向槽为参考-only 未来转向意图 sin/cos(ref_yaw(t+dt+H)-ref_yaw(t+dt))，H=ref_turn_intent_horizon_s。
  2. 进 ONNX 前仅 clip_observations=100；不在 Python 里做 (obs-mean)/std（由 ONNX 图内 Sub/Div 完成）。
  3. ONNX 输出为 raw action；不在 Python 里对 action 做 normalize。
  4. clip_actions=0.6 → 低通 → action_history → action_delay → PD。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from jitter_slip_diagnostics import (
    JitterSlipDiagnostics,
    JitterSlipDiagnosticsConfig,
    _euler_xyz_from_quat_xyzw,
    _quat_rotate_inverse_np,
)
from evaluation_contract import (
    validate_canonical_observation_layout,
    world_ang_vel_to_local_xyzw,
)

PRJ_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPO_ROOT = os.path.dirname(PRJ_DIR)

# C306RMGCfg.env 与 Actor 输入维（须与 ONNX input 第二维一致）
N_MIMIC_OBS = 9 + 21
N_TRACKING_ERROR_OBS = 1 + 3 * 5
N_UPPER_BODY_OBS = 1 + 1 + 2 + 2 * 3
N_PROPRIO_OBS = 3 + 3 + 3 * 21 + N_TRACKING_ERROR_OBS + N_UPPER_BODY_OBS
N_OBS_SINGLE = N_MIMIC_OBS + N_PROPRIO_OBS  # 125
HISTORY_LEN = 10
N_POLICY_OBS = N_OBS_SINGLE * (HISTORY_LEN + 1)  # 1375
REF_TURN_INTENT_HORIZON_S = 0.6  # cp_rmg_config.env.ref_turn_intent_horizon_s
# hmft turn_curriculum.apply_to_substrings — scale ref yaw rate only on these motions
DEFAULT_TURN_YAW_SCALE_MOTION_SUBSTRINGS = ("YuanDiZhuan_You", "YuanDiZhuan_Zuo")

# ---------------------------------------------------------------------------
# C306RMGRealRepairCfg + C306Mimic_V1 内嵌常量（与 cp_rmg_config / cp_fdd_V1 一致）
# ---------------------------------------------------------------------------
CP_RMG_V1: dict = {
    "default_joint_angles": {
        "left_leg_pelvic_pitch_joint": -0.185,
        "left_leg_pelvic_roll_joint": 0,
        "left_leg_pelvic_yaw_joint": 0,
        "left_leg_knee_pitch_joint": 0.36,
        "left_leg_ankle_pitch_joint": -0.175,
        "left_leg_ankle_roll_joint": 0,
        "right_leg_pelvic_pitch_joint": -0.185,
        "right_leg_pelvic_roll_joint": 0,
        "right_leg_pelvic_yaw_joint": 0.0,
        "right_leg_knee_pitch_joint": 0.36,
        "right_leg_ankle_pitch_joint": -0.175,
        "right_leg_ankle_roll_joint": 0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_pitch_joint": -1.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_pitch_joint": -1.0,
    },
    "action_scale": (
        0.25, 0.25, 0.25, 0.25, 0.20, 0.20,
        0.25, 0.25, 0.25, 0.25, 0.20, 0.20,
        0.20,
        0.15, 0.15, 0.15, 0.15,
        0.15, 0.15, 0.15, 0.15,
    ),
    "stiffness": {
        "leg_pelvic_pitch": 175.0,
        "leg_pelvic_roll": 175.0,
        "leg_pelvic_yaw": 100.0,
        "leg_knee": 175.0,
        "leg_ankle_pitch": 100.0,
        "leg_ankle_roll": 100.0,
        "waist": 100.0,
        "shoulder_pitch": 80.0,
        "shoulder_roll": 80.0,
        "shoulder_yaw": 60.0,
        "elbow": 80.0,
    },
    "damping": {
        "leg_pelvic_pitch": 7.0,
        "leg_pelvic_roll": 7.0,
        "leg_pelvic_yaw": 4.0,
        "leg_knee": 7.0,
        "leg_ankle_pitch": 5.0,
        "leg_ankle_roll": 5.0,
        "waist": 4.0,
        "shoulder_pitch": 3.5,
        "shoulder_roll": 3.5,
        "shoulder_yaw": 2.5,
        "elbow": 3.5,
    },
    "decimation": 10,
    "simulation_dt": 0.002,
    "clip_actions": 0.6,
    "clip_observations": 100.0,
    "obs_ang_vel_scale": 0.25,
    "obs_dof_pos_scale": 1.0,
    "obs_dof_vel_scale": 0.05,
    "history_len": 10,
    "mimic_tar_first_step": 1,
    "ref_turn_intent_horizon_s": REF_TURN_INTENT_HORIZON_S,
    "ankle_idx": (4, 5, 10, 11),
    "upper_key_bodies": (
        "head_link_o",
        "left_elbow_pitch_link",
        "left_hand_link_o",
        "right_elbow_pitch_link",
        "right_hand_link_o",
    ),
    "root_height_offset": 0.0,
    "friction_coeffs": 1.0,
    "torque_safety_limit": 0.9,
    "enable_tn_limit": True,
    "urdf_limit_discount": 0.8,
    "power_limit": None,
    "urdf_basename": "CASBOT02_ENCOS_5dof_skeleton_20250904_box.urdf",
    "action_lowpass_filter": True,
    "action_filter_alpha": 0.4,
    "action_filter_alpha_range": (0.35, 0.45),
    "action_delay": True,
    "action_buf_len": 5,
    "ctrl_delay_step_range": 2,
    "reset_vel_factor": 0.8,
    "root_reset_z_offset": 0.0,
    # G1 balance-safe PD target shaping（默认关闭；G1 --v4 等 preset 开启）
    "g1_balance_safe_enable": False,
    "enable_target_offset_clip": False,
    "enable_target_joint_limit_clamp": False,
    "target_joint_limit_margin": 0.0,
    "target_offset_clip": None,
    # 与 cp_rmg_config.rewards / asset 一致，供 inference CSV 足端奖励诊断
    "ankle_2_ground": 0.05,
    "feet_clearance_simple_target": 0.06,
    "feet_clearance_simple_tolerance": 0.04,
    "feet_use_force_contact": True,
    "feet_contact_force_threshold": 5.0,
    "feet_contact_force_k": 1.0,
    "feet_contact_conf_gamma": 1.0,
    "feet_contact_alpha": 10.0,
    "feet_clearance_alpha": 80.0,
    "feet_min_clearance": 0.08,
    "feet_scuff_lambda": 1.0,
    "feet_scuff_height_k": 40.0,
    "termination_roll": 0.95,
    "termination_pitch": 0.9,
    "terminate_after_contacts_on": ("waist_yaw_link",),
}

# Fail closed at the first detected fall.  The first-failure timestamp is
# recorded separately, so survival metrics never include a post-fall grace
# interval.
FALL_TERMINATION_DELAY_S = 0.0

# 非 early-fail 的终止原因（跑满 ref / eval 上限）
_NON_FAIL_TERMINATION_REASONS = frozenset(
    {
        "motion_end",
        "ref_end",
        "ref_hold_success",
        "timeout",
        "max_time",
        "horizon_truncated",
        "",
    }
)


def compute_ref_motion_duration_s(csv_path: str, csv_fps: float) -> float:
    """从 motion CSV 行数与 fps 估计参考轨迹时长（秒）。"""
    df = pd.read_csv(csv_path)
    n = len(df)
    if n <= 1:
        return 0.0
    return float(n - 1) / float(csv_fps)


def _max_residual_rad(clip_actions: float, action_scale: np.ndarray) -> float:
    """clip_actions 为 residual rad 对称上限；逐维 norm clip = clip_actions / scale_i。"""
    scales = np.asarray(action_scale, dtype=np.float64).reshape(-1)
    if scales.size == 0:
        return float(clip_actions)
    return float(np.max(scales * (float(clip_actions) / scales)))

# v1_clean 基线（clip=5、无低通/无 delay），可用 --v1-clean 或 make_rmg_config 覆盖恢复
CP_RMG_V1_CLEAN: dict = {
    **CP_RMG_V1,
    "clip_actions": 5.0,
    "action_lowpass_filter": False,
    "action_filter_alpha": 1.0,
    "action_delay": False,
}

# C306RMGRefTurnIntentAMASSStartupCfg：无 DR / 无低通 / 无 delay，init_state 腿关节为 0
CP_RMG_AMASS_STARTUP: dict = {
    **CP_RMG_V1,
    "clip_actions": 0.6,
    "action_lowpass_filter": False,
    "action_filter_alpha": 1.0,
    "action_delay": False,
    "root_reset_z_offset": 0.05,
    "yaw_rate_err_soft_clip": True,
    "yaw_rate_err_clip": 1.0,
    "default_joint_angles": {
        "left_leg_pelvic_pitch_joint": 0.0,
        "left_leg_pelvic_roll_joint": 0.0,
        "left_leg_pelvic_yaw_joint": 0.0,
        "left_leg_knee_pitch_joint": 0.0,
        "left_leg_ankle_pitch_joint": 0.0,
        "left_leg_ankle_roll_joint": 0.0,
        "right_leg_pelvic_pitch_joint": 0.0,
        "right_leg_pelvic_roll_joint": 0.0,
        "right_leg_pelvic_yaw_joint": 0.0,
        "right_leg_knee_pitch_joint": 0.0,
        "right_leg_ankle_pitch_joint": 0.0,
        "right_leg_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_pitch_joint": -1.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_pitch_joint": -1.0,
    },
}

# HMFT v4/v5/v6 训练 gate：与 hmft_32700_v6_long_2500_hold5.domain_rand / normalization 一致
CP_RMG_HMFT: dict = {
    **CP_RMG_V1,
    "clip_actions": 0.45,
    "action_delay": False,
    "action_lowpass_filter": True,
    "action_filter_alpha": 0.4,
    "action_filter_alpha_range": (0.40, 0.40),
    "action_buf_len": 2,
    "ctrl_delay_step_range": 0,
}

# HMFT v9ft @ 42300：clip/lowpass=0.60 + 下肢 PD scale（对齐 hmft_39500_v9safe C306RMG42300V9ftCfg）
_V9FT_LOWER_LIMB_KP_SCALE = 1.10
_V9FT_LOWER_LIMB_KD_SCALE = (_V9FT_LOWER_LIMB_KP_SCALE ** 0.5) * 1.05
_V9FT_LOWER_LIMB_STIFFNESS_KEYS = ("leg_pelvic_pitch", "leg_knee", "leg_ankle_pitch")
_V9FT_KD_MAX = {
    "leg_pelvic_pitch": 50.0,
    "leg_pelvic_roll": 50.0,
    "leg_pelvic_yaw": 5.0,
    "leg_knee": 50.0,
    "leg_ankle_pitch": 5.0,
    "leg_ankle_roll": 5.0,
}


def _v9ft_scaled_stiffness_damping() -> tuple[dict, dict]:
    stiffness = dict(CP_RMG_V1["stiffness"])
    damping = dict(CP_RMG_V1["damping"])
    for key in _V9FT_LOWER_LIMB_STIFFNESS_KEYS:
        stiffness[key] = float(stiffness[key]) * _V9FT_LOWER_LIMB_KP_SCALE
        damping[key] = min(float(damping[key]) * _V9FT_LOWER_LIMB_KD_SCALE, _V9FT_KD_MAX[key])
    for key, kd_max in _V9FT_KD_MAX.items():
        if key not in _V9FT_LOWER_LIMB_STIFFNESS_KEYS:
            damping[key] = min(float(damping[key]), kd_max)
    return stiffness, damping


_V9FT_STIFFNESS, _V9FT_DAMPING = _v9ft_scaled_stiffness_damping()
CP_RMG_V9FT: dict = {
    **CP_RMG_V1,
    "stiffness": _V9FT_STIFFNESS,
    "damping": _V9FT_DAMPING,
    "clip_actions": 0.60,
    "action_delay": False,
    "action_lowpass_filter": True,
    "action_filter_alpha": 0.60,
    "action_filter_alpha_range": (0.60, 0.60),
}


def _onnx_embedded_norm_fallback(onnx_path: str) -> bool:
    """无 onnx 包时：根据 ObsNormActor 导出特征做保守检测（mean 初始化 + Sub/Div）。"""
    try:
        with open(onnx_path, "rb") as f:
            blob = f.read(2 * 1024 * 1024)
    except OSError:
        return False
    has_mean = (b"\x04mean" in blob) or (b"mean" in blob[: min(len(blob), 400_000)])
    has_sub_div = (b"\x03Sub" in blob) or (b"\x03Div" in blob) or (b"Sub" in blob[:200_000])
    return bool(has_mean and has_sub_div)


def onnx_has_embedded_normalizer(onnx_path: str) -> Optional[bool]:
    """save_cp_stu ObsNormActor：图开头 Sub(input,mean)+Div+Clip，再进 actor。

    Returns:
        True  — 已确认内嵌 normalizer
        False — 确认为纯 actor（无 mean/Sub/Div）
        None  — 无法解析（例如未安装 onnx 且回退检测未命中）
    """
    try:
        import onnx
    except ImportError:
        return True if _onnx_embedded_norm_fallback(onnx_path) else None
    try:
        model = onnx.load(onnx_path)
    except Exception:
        return True if _onnx_embedded_norm_fallback(onnx_path) else None
    if not model.graph.initializer:
        return False
    has_mean = any(init.name == "mean" or init.name.endswith(".mean") for init in model.graph.initializer)
    if not has_mean:
        return False
    for node in model.graph.node[:12]:
        if node.op_type in ("Sub", "Div", "Clip"):
            return True
    return False


def compute_balance_safe_pd_targets(
    ref_dof_pos: np.ndarray,
    action_cmd: np.ndarray,
    action_scale: np.ndarray,
    *,
    enable_target_offset_clip: bool,
    target_offset_clip: Optional[np.ndarray],
    enable_target_joint_limit_clamp: bool,
    dof_pos_limits: Optional[np.ndarray],
    target_joint_limit_margin: float,
) -> np.ndarray:
    """与 cp_fdd_g1._compute_g1_balance_safe_pd_targets 一致（numpy 版）。"""
    ref = np.asarray(ref_dof_pos, dtype=np.float64).reshape(-1)
    cmd = np.asarray(action_cmd, dtype=np.float64).reshape(-1)
    scale = np.asarray(action_scale, dtype=np.float64).reshape(-1)
    target_raw = ref + cmd * scale

    target = target_raw
    if enable_target_offset_clip and target_offset_clip is not None:
        clip = np.asarray(target_offset_clip, dtype=np.float64).reshape(-1)
        offset_raw = target_raw - ref
        offset = np.clip(offset_raw, -clip, clip)
        target = ref + offset

    if enable_target_joint_limit_clamp and dof_pos_limits is not None:
        limits = np.asarray(dof_pos_limits, dtype=np.float64).reshape(-1, 2)
        margin = float(target_joint_limit_margin)
        lower = limits[:, 0] + margin
        upper = limits[:, 1] - margin
        invalid = lower > upper
        if np.any(invalid):
            mid = 0.5 * (limits[:, 0] + limits[:, 1])
            lower = np.where(invalid, mid, lower)
            upper = np.where(invalid, mid, upper)
        target = np.clip(target, lower, upper)

    return target.astype(np.float32, copy=False)


def pd_gains_for_dof_names(dof_names: tuple, stiffness: dict, damping: dict) -> tuple:
    """与 cp_fdd_V1 `_init_buffers` 按关节名子串匹配 PD 增益。"""
    kps, kds = [], []
    for name in dof_names:
        found = False
        for key in stiffness.keys():
            if key in name:
                kps.append(float(stiffness[key]))
                kds.append(float(damping[key]))
                found = True
                break
        if not found:
            kps.append(0.0)
            kds.append(0.0)
    return tuple(kps), tuple(kds)


POLICY_DOF_NAMES = (
    "left_leg_pelvic_pitch_joint",
    "left_leg_pelvic_roll_joint",
    "left_leg_pelvic_yaw_joint",
    "left_leg_knee_pitch_joint",
    "left_leg_ankle_pitch_joint",
    "left_leg_ankle_roll_joint",
    "right_leg_pelvic_pitch_joint",
    "right_leg_pelvic_roll_joint",
    "right_leg_pelvic_yaw_joint",
    "right_leg_knee_pitch_joint",
    "right_leg_ankle_pitch_joint",
    "right_leg_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
)

# 12-DoF 行走腿序（与 POLICY_DOF_NAMES 前 12 项一致）；用于 sim2sim 消融。
ANKLE_ROLL_ACTION_INDICES = [5, 11]
HIP_ROLL_ACTION_INDICES = [1, 7]

OBS_SANITY_SOURCE = "sim2sim"
OBS_SANITY_DEFAULT_TAGS = (
    "neutral",
    "backward",
    "forward",
    "left",
    "right",
)
OBS_SANITY_DEFAULT_MANUAL_POSES_DEG = {
    "neutral": (0.0, 0.0, 0.0),
    "backward": (0.0, -8.0, 0.0),
    "forward": (0.0, 5.0, 0.0),
    "left": (5.0, 0.0, 0.0),
    "right": (-5.0, 0.0, 0.0),
}
OBS_SANITY_UPPER_KEY_BODY_NAMES = (
    "head_link_o",
    "left_elbow_pitch_link",
    "left_hand_link_o",
    "right_elbow_pitch_link",
    "right_hand_link_o",
)


def build_cp_fdd_v1_single_obs_labels() -> list[str]:
    """125 维 student_obs 标签，顺序与 _build_actor_obs / _build_student_obs_sanity 拼接一致。"""
    labels: list[str] = [
        "mimic/root_height",
        "mimic/ref_roll",
        "mimic/ref_pitch",
        "mimic/ref_turn_intent_sin",
        "mimic/ref_turn_intent_cos",
        "mimic/ref_vel_x",
        "mimic/ref_vel_y",
        "mimic/ref_vel_z",
        "mimic/ref_ang_vel_yaw",
    ]
    labels.extend(f"mimic/ref_dof_pos_{i}" for i in range(len(POLICY_DOF_NAMES)))
    labels.extend(
        (
            "proprio/base_ang_vel_x",
            "proprio/base_ang_vel_y",
            "proprio/base_ang_vel_z",
            "proprio/projected_gravity_x",
            "proprio/projected_gravity_y",
            "proprio/projected_gravity_z",
        )
    )
    labels.extend(f"proprio/dof_pos_minus_default_{i}" for i in range(len(POLICY_DOF_NAMES)))
    labels.extend(f"proprio/dof_vel_{i}" for i in range(len(POLICY_DOF_NAMES)))
    labels.extend(f"proprio/last_action_{i}" for i in range(len(POLICY_DOF_NAMES)))
    labels.append("tracking_error/root_ang_vel_err_yaw")
    labels.extend(f"tracking_error/upper_key_joint_err_{i}" for i in range(3 * len(OBS_SANITY_UPPER_KEY_BODY_NAMES)))
    labels.extend(
        (
            "upper_body/waist_pos",
            "upper_body/waist_vel",
            "upper_body/chest_ang_vel_x",
            "upper_body/chest_ang_vel_y",
        )
    )
    labels.extend(f"upper_body/hand_proxy_{i}" for i in range(6))
    assert len(labels) == N_OBS_SINGLE, f"label count {len(labels)} != N_OBS_SINGLE {N_OBS_SINGLE}"
    return labels


def _str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _quat_xyzw_from_rpy_rad(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """与 isaacgym.torch_utils.quat_from_euler_xyz / _euler_xyz_from_quat_xyzw 互逆（xyzw）。"""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    q = np.array([x, y, z, w], dtype=np.float64)
    return (q / (np.linalg.norm(q) + 1e-12)).astype(np.float32)


def _wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """与 calcHeading / euler_from_quaternion 的 yaw 一致（xyzw）。"""
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 积 q1 * q2，匹配 Eigen（xyzw）。"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _apply_yaw_align_xyzw(q_raw_xyzw: np.ndarray, offset: float) -> np.ndarray:
    """世界系 Z 左乘 R_z(offset)，与 C++ hth_yaw_align_to_ref 一致。"""
    half = 0.5 * float(offset)
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64)
    q = _quat_mul_xyzw(qz, q_raw_xyzw.astype(np.float64))
    return (q / np.linalg.norm(q)).astype(np.float32)


def _quat_rotate_forward_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """本体系向量 v 转到世界系；q 为 xyzw。"""
    q_conj = np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]], dtype=np.float64)
    return _quat_rotate_inverse_np(q_conj, v)


def _parse_obs_sanity_manual_poses(spec: str) -> list[dict]:
    cases: list[dict] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid --obs-sanity-manual-poses segment: {part!r}")
        tag, rpy = part.split(":", 1)
        vals = [float(x.strip()) for x in rpy.split(",")]
        if len(vals) != 3:
            raise ValueError(f"pose {part!r} must be roll,pitch,yaw in degrees")
        cases.append(
            {
                "sanity_tag": tag.strip(),
                "roll_deg": vals[0],
                "pitch_deg": vals[1],
                "yaw_deg": vals[2],
            }
        )
    if not cases:
        raise ValueError("empty --obs-sanity-manual-poses")
    return cases


def _default_obs_sanity_manual_cases() -> list[dict]:
    return [
        {
            "sanity_tag": tag,
            "roll_deg": rpy[0],
            "pitch_deg": rpy[1],
            "yaw_deg": rpy[2],
        }
        for tag, rpy in OBS_SANITY_DEFAULT_MANUAL_POSES_DEG.items()
    ]


def _row_get(row, key, default=None):
    if key not in row.index:
        return default
    val = row[key]
    if pd.isna(val):
        return default
    return val


def _collect_dof_vector(row, prefix: str, n: int) -> Optional[np.ndarray]:
    vals = []
    for i in range(n):
        key = f"{prefix}_{i}"
        if key not in row.index:
            return None
        vals.append(float(row[key]))
    return np.asarray(vals, dtype=np.float32)


def _real_log_row_to_case(row: pd.Series, tag: str) -> dict:
    case = {
        "sanity_tag": tag,
        "roll_deg": float(_row_get(row, "imu_roll_deg", 0.0)),
        "pitch_deg": float(_row_get(row, "imu_pitch_deg", 0.0)),
        "yaw_deg": float(_row_get(row, "imu_yaw_deg", 0.0)),
    }
    ref_index = _row_get(row, "ref_index")
    if ref_index is not None:
        case["ref_row"] = int(ref_index)
    for key in (
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
        "base_ang_vel_x",
        "base_ang_vel_y",
        "base_ang_vel_z",
    ):
        val = _row_get(row, key)
        if val is not None:
            case[key] = float(val)
    n = len(POLICY_DOF_NAMES)
    for prefix in ("ref_dof_pos", "dof_pos", "dof_vel", "raw_action", "applied_action", "actually_sent_action"):
        vec = _collect_dof_vector(row, prefix, n)
        if vec is not None:
            case[prefix] = vec
    return case


def load_hth_rl_log_row(csv_path: str, policy_step: int = 0) -> pd.Series:
    """读取 hth_rl_log.csv 指定 policy_step 行；无 policy_step 列则按行号。"""
    df = pd.read_csv(csv_path)
    step = int(policy_step)
    if "policy_step" in df.columns:
        sub = df[df["policy_step"].astype(int) == step]
        if sub.empty:
            raise ValueError(f"{csv_path!r} 中无 policy_step={step} 的行")
        return sub.iloc[0]
    if step < 0 or step >= len(df):
        raise ValueError(f"policy_step/row {step} 超出 {csv_path!r} 行数 {len(df)}")
    return df.iloc[step]


def hth_rl_log_row_to_inject_case(row: pd.Series) -> dict:
    """hth_rl_log 行 → inject-real-pose 用 dict（dof/IMU/ref 时间等）。"""
    n = len(POLICY_DOF_NAMES)
    case: dict = {}
    ref_index = _row_get(row, "ref_index")
    if ref_index is not None:
        case["ref_row"] = int(ref_index)
    ref_time = _row_get(row, "ref_time_s")
    if ref_time is not None:
        case["ref_time_s"] = float(ref_time)
    policy_step = _row_get(row, "policy_step")
    if policy_step is not None:
        case["policy_step"] = int(policy_step)
    for col in ("imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z"):
        val = _row_get(row, col)
        if val is not None:
            case[col] = float(val)
    for prefix in (
        "dof_pos",
        "dof_vel",
        "raw_action",
        "applied_action",
        "actually_sent_action",
        "filtered_action",
    ):
        vec = _collect_dof_vector(row, prefix, n)
        if vec is not None:
            case[prefix] = vec
    return case


def extract_obs_sanity_cases_from_real_log(
    csv_path: str,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    df = pd.read_csv(csv_path)
    if "sanity_tag" in df.columns and tags:
        cases: list[dict] = []
        for tag in tags:
            sub = df[df["sanity_tag"].astype(str) == str(tag)]
            if sub.empty:
                print(f"[obs-sanity] WARN: no rows for sanity_tag={tag!r} in {csv_path}")
                continue
            if "imu_pitch_deg" in sub.columns:
                target = float(sub["imu_pitch_deg"].abs().median())
                idx = (sub["imu_pitch_deg"].abs() - target).abs().idxmin()
            else:
                idx = sub.index[len(sub) // 2]
            cases.append(_real_log_row_to_case(sub.loc[idx], str(tag)))
        if cases:
            return cases

    if "sanity_tag" in df.columns or "imu_pitch_deg" in df.columns:
        cases = []
        if "imu_pitch_deg" in df.columns:
            pitch = df["imu_pitch_deg"].astype(float)
            neg = df[pitch < 0]
            pos = df[pitch > 0]
            if len(neg):
                target = float(neg["imu_pitch_deg"].median())
                cases.append(_real_log_row_to_case(neg.loc[(neg["imu_pitch_deg"] - target).abs().idxmin()], "backward_auto"))
            if len(pos):
                target = float(pos["imu_pitch_deg"].median())
                cases.append(_real_log_row_to_case(pos.loc[(pos["imu_pitch_deg"] - target).abs().idxmin()], "forward_auto"))
            neutral_idx = pitch.abs().idxmin()
            cases.insert(0, _real_log_row_to_case(df.loc[neutral_idx], "neutral_auto"))
        if "imu_roll_deg" in df.columns:
            roll = df["imu_roll_deg"].astype(float)
            cases.append(_real_log_row_to_case(df.loc[roll.idxmax()], "left_or_right_auto_1"))
            cases.append(_real_log_row_to_case(df.loc[roll.idxmin()], "left_or_right_auto_2"))
        if cases:
            return cases

    raise ValueError(
        f"cannot extract obs sanity cases from {csv_path}; provide sanity_tag column or imu_roll/pitch columns"
    )


def resolve_obs_sanity_cases(args: argparse.Namespace) -> list[dict]:
    if args.obs_sanity_real_log:
        tags = None
        if args.obs_sanity_tags:
            tags = [t.strip() for t in args.obs_sanity_tags.split(",") if t.strip()]
        return extract_obs_sanity_cases_from_real_log(args.obs_sanity_real_log, tags=tags)
    if args.obs_sanity_manual_poses:
        return _parse_obs_sanity_manual_poses(args.obs_sanity_manual_poses)
    return _default_obs_sanity_manual_cases()


def obs_sanity_csv_fieldnames(num_actions: int, log_policy_input: bool) -> list[str]:
    fields = [
        "source",
        "sanity_tag",
        "case_index",
        "ref_index",
        "history_fill_mode",
        "imu_quat_w",
        "imu_quat_x",
        "imu_quat_y",
        "imu_quat_z",
        "imu_roll_rad",
        "imu_pitch_rad",
        "imu_yaw_rad",
        "imu_roll_deg",
        "imu_pitch_deg",
        "imu_yaw_deg",
        "projected_gravity_x",
        "projected_gravity_y",
        "projected_gravity_z",
        "base_ang_vel_x",
        "base_ang_vel_y",
        "base_ang_vel_z",
        "ref_root_pos_x",
        "ref_root_pos_y",
        "ref_root_pos_z",
        "ref_roll",
        "ref_pitch",
        "ref_yaw",
        "ref_vel_x",
        "ref_vel_y",
        "ref_vel_z",
        "ref_ang_vel_x",
        "ref_ang_vel_y",
        "ref_ang_vel_z",
        "ref_yaw_future",
        "ref_turn_intent_sin",
        "ref_turn_intent_cos",
    ]
    for prefix in (
        "ref_dof_pos",
        "ref_dof_vel",
        "dof_pos",
        "dof_vel",
        "dof_pos_minus_ref",
        "raw_action",
        "clipped_action",
        "filtered_action",
        "applied_action",
        "actually_sent_action",
        "target_offset",
        "dof_pos_target",
    ):
        fields.extend(f"{prefix}_{i}" for i in range(num_actions))
    fields.extend(f"hth_single_obs_{i}" for i in range(N_OBS_SINGLE))
    if log_policy_input:
        fields.extend(f"hth_policy_input_{i}" for i in range(N_POLICY_OBS))
    return fields


_INFERENCE_LOG_EXTRA_COLUMNS = (
    "base_roll",
    "base_pitch",
    "base_yaw",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    "left_foot_z",
    "right_foot_z",
    "left_foot_vz",
    "right_foot_vz",
    "left_foot_contact_force",
    "right_foot_contact_force",
    "left_foot_contact_bool",
    "right_foot_contact_bool",
    "touchdown_peak_force",
    "touchdown_vertical_velocity",
    "feet_contact_match",
    "feet_stance_slip",
    "base_lin_vel_x",
    "base_lin_vel_y",
    "base_lin_vel_z",
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "episode_done",
    "termination_reason",
    "reward/feet_clearance_simple",
    "reward/feet_contact_matching",
    "reward/feet_swing_clearance",
    "reward/feet_scuffing",
)

_FOOT_BODY_NAMES = ("left_leg_ankle_roll_link", "right_leg_ankle_roll_link")
_GROUND_GEOM_SUBSTRINGS = ("floor", "ground", "terrain")
_REF_SWING_ANKLE_PITCH_THRESH = -0.10


def _inference_log_columns(
    num_actions: int,
    *,
    log_policy_obs: bool = False,
    log_full_policy_input: bool = False,
) -> tuple:
    cols = ["timestamp", "imu_qw", "imu_qx", "imu_qy", "imu_qz", "imu_wx", "imu_wy", "imu_wz"]
    cols += [f"joint_pos_{i}" for i in range(num_actions)]
    cols += [f"joint_vel_{i}" for i in range(num_actions)]
    # action_* kept for backward compatibility; same as applied_action_* (post-clip, post lowpass/delay).
    cols += [f"action_{i}" for i in range(num_actions)]
    cols += [f"raw_action_{i}" for i in range(num_actions)]
    cols += [f"clipped_action_{i}" for i in range(num_actions)]
    cols += [f"applied_action_{i}" for i in range(num_actions)]
    cols += [f"dof_pos_target_{i}" for i in range(num_actions)]
    cols += [
        "ref_root_z",
        "ref_quat_w",
        "ref_quat_x",
        "ref_quat_y",
        "ref_quat_z",
        "ref_vel_x",
        "ref_vel_y",
        "ref_vel_z",
        "ref_ang_vel_x",
        "ref_ang_vel_y",
        "ref_ang_vel_z",
    ]
    cols += [f"ref_dof_pos_{i}" for i in range(num_actions)]
    cols += list(_INFERENCE_LOG_EXTRA_COLUMNS)
    if log_policy_obs:
        cols += [f"hth_single_obs_{i}" for i in range(N_OBS_SINGLE)]
    if log_full_policy_input:
        cols += [f"hth_policy_input_{i}" for i in range(N_POLICY_OBS)]
    return tuple(cols)


def _resolve_foot_and_ground_geoms(model: mujoco.MjModel) -> tuple:
    """返回 (foot_body_ids[2], foot_geom_to_idx, ground_geom_ids, term_body_ids)。"""
    foot_body_ids = []
    for name in _FOOT_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        foot_body_ids.append(int(bid))

    foot_geom_to_idx: dict[int, int] = {}
    for fi, bid in enumerate(foot_body_ids):
        if bid < 0:
            continue
        gadr = int(model.body_geomadr[bid])
        gnum = int(model.body_geomnum[bid])
        for k in range(gnum):
            gid = gadr + k
            if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
                continue
            foot_geom_to_idx[int(gid)] = fi

    ground_ids: set[int] = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        low = name.lower()
        if any(sub in low for sub in _GROUND_GEOM_SUBSTRINGS):
            ground_ids.add(gid)
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            ground_ids.add(gid)
    for literal in ("floor", "ground"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, literal)
        if gid >= 0:
            ground_ids.add(gid)

    term_body_ids: list[int] = []
    for token in CP_RMG_V1.get("terminate_after_contacts_on", ()):
        for bid in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if token in bname:
                term_body_ids.append(int(bid))

    return foot_body_ids, foot_geom_to_idx, sorted(ground_ids), term_body_ids


def _read_feet_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_body_ids: list[int],
    foot_geom_to_idx: dict[int, int],
    ground_geom_ids: list[int],
    *,
    contact_force_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_foot = 2
    foot_z = np.full((n_foot,), np.nan, dtype=np.float64)
    foot_vz = np.full((n_foot,), np.nan, dtype=np.float64)
    foot_vel_xy_sq = np.zeros((n_foot,), dtype=np.float64)
    foot_fn = np.zeros((n_foot,), dtype=np.float64)
    vel6 = np.zeros(6, dtype=np.float64)
    for i, bid in enumerate(foot_body_ids):
        if bid < 0:
            continue
        foot_z[i] = float(data.xpos[bid, 2])
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
        foot_vz[i] = float(vel6[2])
        foot_vel_xy_sq[i] = float(vel6[0] ** 2 + vel6[1] ** 2)

    ground_set = set(ground_geom_ids)
    f6 = np.zeros(6, dtype=np.float64)
    for ci in range(data.ncon):
        con = data.contact[ci]
        g1, g2 = int(con.geom1), int(con.geom2)
        foot_idx = None
        if g1 in foot_geom_to_idx and g2 in ground_set:
            foot_idx = foot_geom_to_idx[g1]
        elif g2 in foot_geom_to_idx and g1 in ground_set:
            foot_idx = foot_geom_to_idx[g2]
        if foot_idx is None:
            continue
        mujoco.mj_contactForce(model, data, ci, f6)
        foot_fn[foot_idx] += max(float(f6[0]), 0.0)

    foot_contact_bool = foot_fn > float(contact_force_threshold)
    return foot_z, foot_vz, foot_fn, foot_contact_bool, foot_vel_xy_sq


def _estimate_ref_feet_contact_prob(dof_pos_ref: np.ndarray, dof_names: tuple) -> np.ndarray:
    """CSV 无 pkl 足接触时：踝 pitch 高于阈值视为摆动相（与 cp_fdd 默认 0.999 对照）。"""
    lap = float(dof_pos_ref[dof_names.index("left_leg_ankle_pitch_joint")])
    rap = float(dof_pos_ref[dof_names.index("right_leg_ankle_pitch_joint")])

    def _prob(ap: float) -> float:
        return 0.001 if ap > _REF_SWING_ANKLE_PITCH_THRESH else 0.999

    return np.array([_prob(lap), _prob(rap)], dtype=np.float64)


def _compute_feet_reward_terms(
    foot_z: np.ndarray,
    foot_fn: np.ndarray,
    foot_vel_xy_sq: np.ndarray,
    ref_contact_prob: np.ndarray,
    *,
    ankle_2_ground: float,
    feet_clearance_simple_target: float,
    feet_clearance_simple_tolerance: float,
    feet_use_force_contact: bool,
    feet_contact_force_threshold: float,
    feet_contact_force_k: float,
    feet_contact_conf_gamma: float,
    feet_contact_alpha: float,
    feet_clearance_alpha: float,
    feet_min_clearance: float,
    feet_scuff_lambda: float,
    feet_scuff_height_k: float,
) -> dict[str, float]:
    ref_swing = np.clip((0.999 - ref_contact_prob) / (0.999 - 0.001), 0.0, 1.0)
    swing_sum = float(np.clip(ref_swing.sum(), 1e-6, None))
    swing_active = swing_sum > 1e-4

    if feet_use_force_contact:
        sim_contact_prob = 1.0 / (1.0 + np.exp(-feet_contact_force_k * (foot_fn - feet_contact_force_threshold)))
    else:
        sim_contact_prob = (foot_fn > feet_contact_force_threshold).astype(np.float64)

    contact_conf = np.abs(2.0 * ref_contact_prob - 1.0) ** feet_contact_conf_gamma
    contact_error = contact_conf * (sim_contact_prob - ref_contact_prob) ** 2
    r_contact = float(np.exp(-feet_contact_alpha * contact_error.sum()))

    foot_sole = foot_z - ankle_2_ground
    h_target = np.full_like(foot_sole, feet_min_clearance)
    height_error = np.clip(h_target - foot_sole, 0.0, None)
    r_clear_each = ref_swing * np.exp(-feet_clearance_alpha * height_error**2)
    r_swing_clear = float(r_clear_each.sum())

    low_height = 1.0 / (1.0 + np.exp(-feet_scuff_height_k * (h_target - foot_sole)))
    r_scuff = float(-feet_scuff_lambda * np.sum(ref_swing * low_height * foot_vel_xy_sq))

    sole_clearance = np.clip(foot_sole, 0.0, 0.20)
    clearance_reward = np.exp(-np.abs(sole_clearance - feet_clearance_simple_target) / max(feet_clearance_simple_tolerance, 1e-6))
    if swing_active:
        r_clearance_simple = float(np.sum(ref_swing * clearance_reward) / swing_sum)
    else:
        r_clearance_simple = 0.0

    return {
        "reward/feet_clearance_simple": r_clearance_simple,
        "reward/feet_contact_matching": r_contact,
        "reward/feet_swing_clearance": r_swing_clear,
        "reward/feet_scuffing": r_scuff,
    }


def _termination_contact_on_bad_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    term_body_ids: list[int],
    ground_geom_ids: list[int],
) -> bool:
    if not term_body_ids or not ground_geom_ids:
        return False
    term_set = set(term_body_ids)
    ground_set = set(ground_geom_ids)
    f6 = np.zeros(6, dtype=np.float64)
    for ci in range(data.ncon):
        con = data.contact[ci]
        g1, g2 = int(con.geom1), int(con.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        hit = (b1 in term_set and g2 in ground_set) or (b2 in term_set and g1 in ground_set)
        if not hit:
            continue
        mujoco.mj_contactForce(model, data, ci, f6)
        if float(np.linalg.norm(f6[:3])) > 1.0:
            return True
    return False


def quat_conjugate_torch(q):
    out = q.clone()
    out[..., :3] = -out[..., :3]
    return out


def quat_mul_torch(q, r):
    x1, y1, z1, w1 = q.unbind(-1)
    x2, y2, z2, w2 = r.unbind(-1)
    return torch.stack((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ), dim=-1)


def quat_diff(q0, q1):
    return quat_mul_torch(q1, quat_conjugate_torch(q0))


def quat_to_exp_map(q):
    q = q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)
    sin_theta = torch.sqrt(torch.clamp(1.0 - q[..., 3] * q[..., 3], min=0.0))
    angle = wrap_to_pi_torch(2.0 * torch.acos(torch.clamp(q[..., 3], -1.0, 1.0)))
    axis = q[..., :3] / torch.clamp(sin_theta.unsqueeze(-1), min=1e-8)
    mask = torch.abs(sin_theta) > 1e-5
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1.0
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    axis = torch.where(mask.unsqueeze(-1), axis, default_axis)
    return angle.unsqueeze(-1) * axis


def slerp(q0, q1, blend):
    q0 = q0 / torch.clamp(torch.norm(q0, dim=-1, keepdim=True), min=1e-8)
    q1 = q1 / torch.clamp(torch.norm(q1, dim=-1, keepdim=True), min=1e-8)
    cos_half_theta = torch.sum(q0 * q1, dim=-1)
    q1 = torch.where((cos_half_theta < 0.0).unsqueeze(-1), -q1, q1)
    cos_half_theta = torch.abs(cos_half_theta).clamp(-1.0, 1.0).unsqueeze(-1)
    blend = blend.unsqueeze(-1)
    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(torch.clamp(1.0 - cos_half_theta * cos_half_theta, min=0.0))
    ratio_a = torch.sin((1.0 - blend) * half_theta) / torch.clamp(sin_half_theta, min=1e-8)
    ratio_b = torch.sin(blend * half_theta) / torch.clamp(sin_half_theta, min=1e-8)
    new_q = ratio_a * q0 + ratio_b * q1
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    return torch.where(torch.abs(cos_half_theta) >= 1.0, q0, new_q)


def _smooth(x, box_pts, device):
    box = torch.ones(box_pts, device=device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding="same",
    )
    return smoothed.squeeze(0).T


class CsvMotionLib:
    def __init__(self, csv_file, device, fps=50.0):
        self._csv_path = os.path.abspath(csv_file)
        self._device = device
        self._body_link_list = [
            "left_leg_knee_pitch_link",
            "left_leg_ankle_roll_link",
            "right_leg_knee_pitch_link",
            "right_leg_ankle_roll_link",
            "head_link_o",
            "left_elbow_pitch_link",
            "left_hand_link_o",
            "right_elbow_pitch_link",
            "right_hand_link_o",
        ]
        self._load_csv(csv_file, fps)

    def _load_csv(self, csv_file, fps):
        df = pd.read_csv(csv_file)
        req = ["root_x", "root_y", "root_z", "root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"CSV 缺少必要列: {miss}")
        dof_cols = sorted([c for c in df.columns if c.startswith("dof_pos_")], key=lambda x: int(x.split("_")[-1]))
        if len(dof_cols) == 0:
            raise ValueError("CSV 中未找到 dof_pos_* 列")

        self._motion_fps = torch.tensor([float(fps)], dtype=torch.float, device=self._device)
        self._motion_dt = 1.0 / self._motion_fps
        root_pos = torch.tensor(df[["root_x", "root_y", "root_z"]].to_numpy(), dtype=torch.float, device=self._device)
        root_rot = torch.tensor(df[["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"]].to_numpy(), dtype=torch.float, device=self._device)
        root_rot = root_rot / torch.clamp(torch.norm(root_rot, dim=1, keepdim=True), min=1e-8)
        dof_pos = torch.tensor(df[dof_cols].to_numpy(), dtype=torch.float, device=self._device)
        local_body_pos = torch.zeros((root_pos.shape[0], len(self._body_link_list), 3), dtype=torch.float, device=self._device)

        num_frames = root_pos.shape[0]
        self._motion_num_frames = torch.tensor([num_frames], dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor([(num_frames - 1) / float(fps)], dtype=torch.float, device=self._device)
        self._motion_root_pos_delta = (root_pos[-1] - root_pos[0]).unsqueeze(0)
        self._motion_root_pos_delta[..., -1] = 0.0

        root_vel = torch.zeros_like(root_pos)
        root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
        root_vel[-1, :] = root_vel[-2, :]
        root_vel = _smooth(root_vel, 19, self._device)

        root_ang_vel = torch.zeros_like(root_pos)
        root_drot = quat_diff(root_rot[:-1], root_rot[1:])
        root_ang_vel[:-1, :] = fps * quat_to_exp_map(root_drot)
        root_ang_vel[-1, :] = root_ang_vel[-2, :]
        root_ang_vel = _smooth(root_ang_vel, 19, self._device)

        dof_vel = torch.zeros_like(dof_pos)
        dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
        dof_vel[-1, :] = dof_vel[-2, :]
        dof_vel = _smooth(dof_vel, 19, self._device)

        self._motion_root_pos = root_pos
        self._motion_root_rot = root_rot
        self._motion_root_vel = root_vel
        self._motion_root_ang_vel = root_ang_vel
        self._motion_dof_pos = dof_pos
        self._motion_dof_vel = dof_vel
        self._motion_local_body_pos = local_body_pos
        self._motion_weights = torch.tensor([1.0], dtype=torch.float, device=self._device)
        self._motion_ids = torch.tensor([0], dtype=torch.long, device=self._device)

    def sample_motions(self, n):
        return torch.zeros((n,), dtype=torch.long, device=self._device)

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_key_body_idx(self, key_body_names):
        return [self._body_link_list.index(name) for name in key_body_names]

    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]
        phase = times / self._motion_lengths[motion_ids]
        phase = torch.clip(phase, 0.0, 1.0)
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0.float()
        return frame_idx0, frame_idx1, blend

    def calc_motion_frame(self, motion_ids, motion_times):
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times = motion_times - motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

        root_pos0 = self._motion_root_pos[frame_idx0]
        root_pos1 = self._motion_root_pos[frame_idx1]
        root_rot0 = self._motion_root_rot[frame_idx0]
        root_rot1 = self._motion_root_rot[frame_idx1]
        root_vel = self._motion_root_vel[frame_idx0]
        root_ang_vel = self._motion_root_ang_vel[frame_idx0]
        dof_pos0 = self._motion_dof_pos[frame_idx0]
        dof_pos1 = self._motion_dof_pos[frame_idx1]
        local_key_body_pos0 = self._motion_local_body_pos[frame_idx0]
        local_key_body_pos1 = self._motion_local_body_pos[frame_idx1]
        dof_vel = self._motion_dof_vel[frame_idx0]

        blend_u = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_u) * root_pos0 + blend_u * root_pos1
        root_pos += motion_loop_num.unsqueeze(-1) * self._motion_root_pos_delta[motion_ids]
        root_rot = slerp(root_rot0, root_rot1, blend)
        dof_pos = (1.0 - blend_u) * dof_pos0 + blend_u * dof_pos1
        local_key_body_pos = (1.0 - blend_u.unsqueeze(1)) * local_key_body_pos0 + blend_u.unsqueeze(1) * local_key_body_pos1
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def wrap_to_pi_torch(angles):
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def _soft_clamp_obs(x: torch.Tensor, limit: float) -> torch.Tensor:
    """与 cp_rmg_env._soft_clamp 一致：limit * tanh(x / limit)。"""
    lim = max(float(limit), 1e-6)
    return lim * torch.tanh(x / lim)


def ref_turn_intent_heading_sin_cos(root_rot_now: torch.Tensor, root_rot_future: torch.Tensor) -> tuple:
    """sin/cos(ref_yaw(t+H)-ref_yaw(t))，与 cp_fdd_V1._compute_ref_turn_intent_heading_feature 一致。"""
    q_rel = quat_mul_torch(quat_conjugate_torch(root_rot_now), root_rot_future)
    _, _, ref_turn_delta = euler_from_quaternion(q_rel)
    ref_turn_delta = wrap_to_pi_torch(ref_turn_delta)
    return torch.sin(ref_turn_delta).unsqueeze(-1), torch.cos(ref_turn_delta).unsqueeze(-1)


def euler_from_quaternion(quat_angle):
    if quat_angle.dim() == 1:
        quat_angle = quat_angle.unsqueeze(0)
    quat_angle = quat_angle / torch.norm(quat_angle, dim=1, keepdim=True)
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


@dataclass
class RMGConfig:
    device: str
    simulation_dt: float
    decimation: int
    sim_duration: Optional[float]
    clip_observations: float
    clip_actions: float
    friction_coeffs: float
    obs_ang_vel_scale: float
    obs_dof_pos_scale: float
    obs_dof_vel_scale: float
    history_len: int
    num_actions: int
    action_scale: tuple
    dof_names: tuple
    default_dof_pos: tuple
    kps: tuple
    kds: tuple
    upper_key_bodies: tuple
    ankle_idx: tuple
    root_height_offset: float
    root_reset_z_offset: float
    mimic_tar_first_step: int
    ref_turn_intent_horizon_s: float
    reset_vel_factor: float
    zero_policy_action: bool
    zero_ankle_roll_action: bool
    zero_hip_roll_and_ankle_roll_action: bool
    torque_safety_limit: float
    enable_tn_limit: bool
    urdf_limit_discount: float
    power_limit: Optional[tuple]
    urdf_path: str
    action_lowpass_filter: bool
    action_filter_alpha: float
    action_delay: bool
    action_buf_len: int
    ctrl_delay_step_range: int
    action_delay_steps: Optional[int]
    ankle_pitch_bias_left: float
    ankle_pitch_bias_right: float
    inference_log_csv: Optional[str]
    inference_log_policy_obs: bool
    inference_log_full_policy_input: bool
    g1_balance_safe_enable: bool
    enable_target_offset_clip: bool
    enable_target_joint_limit_clamp: bool
    target_joint_limit_margin: float
    target_offset_clip: Optional[tuple]
    yaw_rate_err_soft_clip: bool
    yaw_rate_err_clip: float
    # 真机 hth_yaw_align_to_ref：obs 层 yaw 欺骗
    yaw_align_to_ref: bool
    yaw_align_max_deg: float
    # True：base_ang_vel / tracking_err / upper_body ang_vel 也用 q_policy（sim 完全忽视 yaw）
    # False：与 C++ 一致，仅 projected_gravity 用 q_policy
    yaw_align_full_proprio: bool
    # True：reset 时随机物理 yaw（sim 压力测试）；False：只改 obs，不动 MuJoCo 朝向
    yaw_align_random_yaw: bool
    obs_inject_real_log: Optional[str]
    obs_inject_upper_error: bool
    obs_inject_upper_vel: bool
    obs_inject_upper_scale: float
    obs_inject_time_col: str
    obs_inject_time_shift_s: float


def make_rmg_config(
    device: str = "cpu",
    sim_duration: Optional[float] = None,
    reset_vel_factor: Optional[float] = None,
    v1_clean: bool = False,
    hmft: bool = False,
    v9ft: bool = False,
    amass_startup: bool = False,
    **overrides,
) -> RMGConfig:
    """由 CP_RMG_V1（或 v1_clean / hmft / v9ft / amass_startup 预设）构造 RMGConfig；overrides 可覆盖任意 dataclass 字段。"""
    if v1_clean:
        p = CP_RMG_V1_CLEAN
    elif hmft:
        p = CP_RMG_HMFT
    elif v9ft:
        p = CP_RMG_V9FT
    elif amass_startup:
        p = CP_RMG_AMASS_STARTUP
    else:
        p = CP_RMG_V1
    dof_names = POLICY_DOF_NAMES
    n = len(dof_names)
    if len(p["action_scale"]) != n:
        raise ValueError(f"action_scale len {len(p['action_scale'])} != {n}")
    missing = [j for j in dof_names if j not in p["default_joint_angles"]]
    if missing:
        raise ValueError(f"default_joint_angles missing: {missing}")
    default_dof_pos = tuple(float(p["default_joint_angles"][j]) for j in dof_names)
    kps, kds = pd_gains_for_dof_names(dof_names, p["stiffness"], p["damping"])
    urdf_path = os.path.join(REPO_ROOT, "assets", "casbot_02", "urdf", p["urdf_basename"])
    cfg = RMGConfig(
        device=device,
        simulation_dt=p["simulation_dt"],
        decimation=p["decimation"],
        sim_duration=sim_duration,
        clip_observations=p["clip_observations"],
        clip_actions=p["clip_actions"],
        friction_coeffs=p["friction_coeffs"],
        obs_ang_vel_scale=p["obs_ang_vel_scale"],
        obs_dof_pos_scale=p["obs_dof_pos_scale"],
        obs_dof_vel_scale=p["obs_dof_vel_scale"],
        history_len=p["history_len"],
        num_actions=n,
        action_scale=p["action_scale"],
        dof_names=dof_names,
        default_dof_pos=default_dof_pos,
        kps=kps,
        kds=kds,
        upper_key_bodies=p["upper_key_bodies"],
        ankle_idx=p["ankle_idx"],
        root_height_offset=p["root_height_offset"],
        root_reset_z_offset=p["root_reset_z_offset"],
        mimic_tar_first_step=p["mimic_tar_first_step"],
        ref_turn_intent_horizon_s=float(p["ref_turn_intent_horizon_s"]),
        reset_vel_factor=reset_vel_factor if reset_vel_factor is not None else p["reset_vel_factor"],
        zero_policy_action=False,
        zero_ankle_roll_action=False,
        zero_hip_roll_and_ankle_roll_action=False,
        torque_safety_limit=p["torque_safety_limit"],
        enable_tn_limit=p["enable_tn_limit"],
        urdf_limit_discount=p["urdf_limit_discount"],
        power_limit=p["power_limit"],
        urdf_path=urdf_path,
        action_lowpass_filter=p["action_lowpass_filter"],
        action_filter_alpha=p["action_filter_alpha"],
        action_delay=p["action_delay"],
        action_buf_len=int(p["action_buf_len"]),
        ctrl_delay_step_range=int(p["ctrl_delay_step_range"]),
        action_delay_steps=None,
        ankle_pitch_bias_left=0.0,
        ankle_pitch_bias_right=0.0,
        inference_log_csv=None,
        inference_log_policy_obs=False,
        inference_log_full_policy_input=False,
        g1_balance_safe_enable=bool(p.get("g1_balance_safe_enable", False)),
        enable_target_offset_clip=bool(p.get("enable_target_offset_clip", False)),
        enable_target_joint_limit_clamp=bool(p.get("enable_target_joint_limit_clamp", False)),
        target_joint_limit_margin=float(p.get("target_joint_limit_margin", 0.0)),
        target_offset_clip=p.get("target_offset_clip"),
        yaw_rate_err_soft_clip=bool(p.get("yaw_rate_err_soft_clip", False)),
        yaw_rate_err_clip=float(p.get("yaw_rate_err_clip", 1.0)),
        yaw_align_to_ref=False,
        yaw_align_max_deg=180.0,
        yaw_align_full_proprio=True,
        yaw_align_random_yaw=False,
        obs_inject_real_log=None,
        obs_inject_upper_error=False,
        obs_inject_upper_vel=False,
        obs_inject_upper_scale=1.0,
        obs_inject_time_col="auto",
        obs_inject_time_shift_s=0.0,
    )
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"unknown RMGConfig field: {key}")
        setattr(cfg, key, val)
    return cfg


def _validate_deploy_alignment(cfg: RMGConfig, onnx_obs_dim: int, onnx_has_norm: Optional[bool]) -> None:
    if cfg.history_len != HISTORY_LEN:
        raise ValueError(f"history_len must be {HISTORY_LEN}, got {cfg.history_len}")
    if onnx_obs_dim != N_POLICY_OBS:
        raise ValueError(f"ONNX input dim must be {N_POLICY_OBS}, got {onnx_obs_dim}")
    if abs(cfg.clip_observations - 100.0) > 1e-6:
        raise ValueError(f"clip_observations must be 100.0 for training/ONNX alignment, got {cfg.clip_observations}")
    if onnx_has_norm is False:
        raise ValueError(
            "ONNX 未检测到内嵌 normalizer（无 mean+Sub/Div）。请用 save_cp_stu.py --with_normalizer 导出，"
            "或在纯 actor ONNX 上自行实现与 checkpoint 一致的 normalize；本脚本不在 Python 侧做归一化。"
        )
    if onnx_has_norm is None:
        print(
            "[deploy] WARN: 无法解析 ONNX 图（建议 pip install onnx）；"
            "假定已内嵌 normalizer，请勿在 Python 侧重复 normalize。"
        )
    if not cfg.action_delay:
        delay_desc = "off"
    elif cfg.action_delay_steps is not None:
        delay_desc = str(cfg.action_delay_steps)
    else:
        delay_desc = f"rand 0..{cfg.ctrl_delay_step_range}"
    norm_desc = "embedded" if onnx_has_norm else "assumed-embedded"
    action_scale = np.asarray(cfg.action_scale, dtype=np.float64)
    max_res = _max_residual_rad(float(cfg.clip_actions), action_scale)
    norm_clip_max = float(np.max(float(cfg.clip_actions) / action_scale))
    print(
        "[deploy] obs: raw 1375 = 125×(1+10); mimic heading=ref_turn_intent "
        f"(H={cfg.ref_turn_intent_horizon_s}s); clip_obs=100 before ONNX; "
        f"ONNX normalizer={norm_desc}; actions: residual_clip={cfg.clip_actions} rad "
        f"(norm_clip_max={norm_clip_max:.4f}, action_scale_max={float(np.max(action_scale)):.4f}, "
        f"effective_max_residual={max_res:.4f} rad); "
        f"lowpass={cfg.action_lowpass_filter}(alpha={cfg.action_filter_alpha}) "
        f"delay={cfg.action_delay}(steps={delay_desc}) "
        f"balance_safe={cfg.g1_balance_safe_enable} "
        f"yaw_align_to_ref={cfg.yaw_align_to_ref}"
        + (
            f"(full_proprio={cfg.yaw_align_full_proprio}, random_yaw={cfg.yaw_align_random_yaw}, "
            f"max_deg={cfg.yaw_align_max_deg:.0f})"
            if cfg.yaw_align_to_ref
            else ""
        )
        )


def validate_canonical_runner_observation_layout(
    labels,
    scales,
    *,
    single_dim: int,
    history_len: int,
    policy_dim: int,
    ankle_velocity_indices,
):
    """Expose the dependency-free ABI check to G1's patched runner."""
    return validate_canonical_observation_layout(
        labels,
        scales,
        single_dim=single_dim,
        history_len=history_len,
        policy_dim=policy_dim,
        ankle_velocity_indices=ankle_velocity_indices,
    )


class MujocoRMGSim:
    def __init__(
        self,
        xml_path,
        onnx_path,
        motion_csv,
        csv_fps,
        cfg: RMGConfig,
        skip_onnx_normalizer_check: bool = False,
    ):
        self.cfg = cfg
        self.onnx_path = os.path.abspath(onnx_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.simulation_dt
        self.dt = cfg.simulation_dt * cfg.decimation
        self.device = torch.device(cfg.device)
        if skip_onnx_normalizer_check:
            self._onnx_has_normalizer = True
        else:
            detected = onnx_has_embedded_normalizer(self.onnx_path)
            self._onnx_has_normalizer = True if detected is None else detected
        self.policy = onnxruntime.InferenceSession(self.onnx_path)
        self.policy_input_name = self.policy.get_inputs()[0].name
        self.policy_output_name = self.policy.get_outputs()[0].name
        self.onnx_obs_dim = int(self.policy.get_inputs()[0].shape[1])
        if self.onnx_obs_dim % (cfg.history_len + 1) != 0:
            raise ValueError(f"Unexpected ONNX input dim {self.onnx_obs_dim}")
        self.single_obs_dim = self.onnx_obs_dim // (cfg.history_len + 1)
        if self.single_obs_dim != N_OBS_SINGLE:
            raise ValueError(f"single obs dim must be {N_OBS_SINGLE}, got {self.single_obs_dim}")
        _validate_deploy_alignment(cfg, self.onnx_obs_dim, self._onnx_has_normalizer)
        self.action_scale = np.asarray(cfg.action_scale, dtype=np.float32)
        self.default_dof_pos = np.asarray(cfg.default_dof_pos, dtype=np.float32)
        self.default_dof_pos_t = torch.tensor(self.default_dof_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.kps_policy = np.asarray(cfg.kps, dtype=np.float32)
        self.kds_policy = np.asarray(cfg.kds, dtype=np.float32)
        self.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device)
        self.action_tensor = torch.zeros((1, cfg.num_actions), dtype=torch.float32, device=self.device)
        self.obs_history_buf = torch.zeros((1, cfg.history_len, self.single_obs_dim), dtype=torch.float32, device=self.device)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self.ref_dof_pos_res = np.zeros((cfg.num_actions,), dtype=np.float32)
        self.filtered_actions = np.zeros((cfg.num_actions,), dtype=np.float32)
        self.action_history_buf = np.zeros((cfg.action_buf_len, cfg.num_actions), dtype=np.float32)
        self._action_delay_steps = 0
        if cfg.target_offset_clip is not None:
            self._target_offset_clip = np.asarray(cfg.target_offset_clip, dtype=np.float32).reshape(-1)
            if self._target_offset_clip.size != cfg.num_actions:
                raise ValueError(
                    f"target_offset_clip length ({self._target_offset_clip.size}) "
                    f"must equal num_actions ({cfg.num_actions})"
                )
        else:
            self._target_offset_clip = None
        self.dof_pos_limits_policy: Optional[np.ndarray] = None
        self.motion_time_offset = 0.0
        self._build_dof_mappings()
        self._build_body_mappings()
        self.motion_lib = CsvMotionLib(motion_csv, device=str(self.device), fps=csv_fps)
        self._motion_csv_path = os.path.abspath(str(motion_csv))
        self.turn_yaw_scale: float = 1.0
        self._turn_yaw_scale_motion_substrings: tuple[str, ...] = DEFAULT_TURN_YAW_SCALE_MOTION_SUBSTRINGS
        self.mimic_ref_wz_scaled: bool = False
        self.ref_pose_scaled: bool = False
        self.ref_end_hold_enabled: bool = False
        self.ref_end_hold_s: float = 0.0
        self.zero_ref_vel_after_end: bool = True
        self.reset_after_hold_if_alive: bool = True
        self.max_episode_length_s: float | None = None
        self.motion_ids = self.motion_lib.sample_motions(1)
        self.motion_len = float(self.motion_lib.get_motion_length(self.motion_ids)[0].item())
        self._configure_model()
        self._jitter_diag: Optional[JitterSlipDiagnostics] = None
        self._init_yaw_diag: Optional[float] = None
        (
            self._foot_body_ids,
            self._foot_geom_to_idx,
            self._ground_geom_ids,
            self._term_body_ids,
        ) = _resolve_foot_and_ground_geoms(self.model)
        self._prev_foot_contact_bool = [False, False]
        self._fall_detected_motion_time: float | None = None
        self._first_failure_time_s: float | None = None
        self._first_failure_reason: str = ""
        self.full_ref: bool = False
        self.ref_duration_s: float = float(self.motion_len)
        self.eval_duration_s: float = float(self.motion_len)
        self.csv_fps: float = float(csv_fps)
        self.run_metadata_json: str | None = None
        self._run_stop_reason: str = ""
        self._last_motion_time: float = 0.0
        self._yaw_align_offset: float = 0.0
        self._yaw_align_capture_pending: bool = False
        self._robot_yaw_raw: float = 0.0
        self._inject_real_pose_csv: str | None = None
        self._inject_real_pose_step: int = 0
        self._inject_real_pose_use_imu: bool = True
        self._inject_real_pose_use_dof_vel: bool = False
        self._inject_real_pose_last_action: bool = False
        self._inject_real_pose_zero_vel: bool = True
        self._inference_log_fp = None
        self._inference_log_writer = None
        if cfg.inference_log_csv:
            log_path = os.path.abspath(cfg.inference_log_csv)
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._inference_log_fp = open(log_path, "w", newline="", encoding="utf-8")
            self._inference_log_writer = csv.writer(self._inference_log_fp)
            self._inference_log_writer.writerow(
                _inference_log_columns(
                    cfg.num_actions,
                    log_policy_obs=bool(cfg.inference_log_policy_obs),
                    log_full_policy_input=bool(cfg.inference_log_full_policy_input),
                )
            )

    def _close_inference_log(self):
        if self._inference_log_fp is not None:
            self._inference_log_fp.close()
            self._inference_log_fp = None
            self._inference_log_writer = None

    def _fall_termination_reason(self) -> tuple[bool, str]:
        """倒地判定：与 cp_fdd_V1.check_termination 中 roll/pitch/腰触地一致（不含 motion_end）。"""
        base_quat = self._get_base_quat_xyzw().astype(np.float64)
        roll, pitch, _ = _euler_xyz_from_quat_xyzw(base_quat)
        if abs(roll) > float(CP_RMG_V1["termination_roll"]):
            return True, "roll"
        if abs(pitch) > float(CP_RMG_V1["termination_pitch"]):
            return True, "pitch"
        if _termination_contact_on_bad_body(
            self.model, self.data, self._term_body_ids, self._ground_geom_ids
        ):
            return True, "contact_force"
        return False, ""

    def _effective_eval_duration_s(self) -> float:
        if self.cfg.sim_duration is not None:
            return float(self.cfg.sim_duration)
        return float(self.eval_duration_s)

    def _episode_termination_flags(self, unclamped_motion_time: float) -> tuple[int, str]:
        """与 cp_fdd_V1.check_termination 对齐的启发式（单 env sim2sim）。"""
        u_t = float(unclamped_motion_time)
        fallen, reason = self._fall_termination_reason()
        if fallen:
            if self._first_failure_time_s is None:
                self._first_failure_time_s = u_t - float(self.motion_time_offset)
                self._first_failure_reason = reason or "fall"
            return 1, reason or "fall"
        playback_end_t = self._motion_playback_end_time() - self.dt - 1e-6
        ref_end_t = float(self.ref_duration_s) - self.dt - 1e-6
        motion_end_t = float(self.motion_len) - self.dt - 1e-6
        if self.ref_end_hold_enabled:
            past_ref = self._motion_time_past_ref_end(u_t)
            if past_ref >= float(self.ref_end_hold_s) - 1e-6 and not fallen:
                return 1, "ref_hold_success"
            if u_t >= playback_end_t:
                if u_t < ref_end_t - 1e-4:
                    return 1, "horizon_truncated"
                if past_ref < float(self.ref_end_hold_s) - 1e-6:
                    return 1, "horizon_truncated"
            return 0, ""
        if u_t >= motion_end_t:
            return 1, "motion_end"
        if u_t >= ref_end_t:
            return 1, "ref_end"
        if (
            self.cfg.sim_duration is not None
            and u_t >= playback_end_t
            and u_t < ref_end_t
        ):
            return 1, "max_time"
        return 0, ""

    def _should_stop_run_after_fall(self, unclamped_motion_time: float) -> bool:
        """倒地后累计 FALL_TERMINATION_DELAY_S 仿真时间则结束 run()。"""
        fallen, reason = self._fall_termination_reason()
        if not fallen:
            self._fall_detected_motion_time = None
            return False
        u_t = float(unclamped_motion_time)
        if self._first_failure_time_s is None:
            self._first_failure_time_s = u_t - float(self.motion_time_offset)
            self._first_failure_reason = reason or "fall"
        if self._fall_detected_motion_time is None:
            self._fall_detected_motion_time = u_t
            return float(FALL_TERMINATION_DELAY_S) <= 0.0
        return (u_t - self._fall_detected_motion_time) >= float(FALL_TERMINATION_DELAY_S)

    def _motion_playback_end_time(self) -> float:
        """本次 run 的 motion_time 上界（秒，沿参考轨迹时间轴）。"""
        if self.ref_end_hold_enabled:
            global_cap = (
                float(self.max_episode_length_s)
                if self.max_episode_length_s is not None
                else float(self.motion_len)
            )
            short_end = float(self.ref_duration_s) + float(self.ref_end_hold_s)
            effective = min(global_cap, short_end)
            if float(self.ref_duration_s) + 1e-3 >= global_cap:
                effective = global_cap
            return float(self.motion_time_offset) + effective
        end_t = self.motion_time_offset + self._effective_eval_duration_s()
        return min(float(end_t), float(self.motion_len))

    def _should_continue_run(self, u_t: float, viewer_running: bool) -> bool:
        if not viewer_running:
            return False
        return float(u_t) < self._motion_playback_end_time() - 1e-9

    def _print_run_stop_reason(self, u_t: float) -> None:
        end_t = self._motion_playback_end_time()
        u_t = float(u_t)
        if self._should_stop_run_after_fall(u_t):
            return
        if self.cfg.sim_duration is not None and u_t >= end_t - self.dt - 1e-6:
            if self.ref_end_hold_enabled:
                planned_end = float(self.motion_time_offset) + float(self.ref_duration_s) + float(
                    self.ref_end_hold_s
                )
                completed_hold = end_t >= planned_end - self.dt - 1e-6
                self._run_stop_reason = "ref_hold_success" if completed_hold else "horizon_truncated"
                print(
                    f"[sim2sim] {'terminal hold complete' if completed_hold else 'horizon truncated'} "
                    f"(motion_time={u_t:.2f}, effective_horizon={end_t - self.motion_time_offset:.2f}s)",
                    flush=True,
                )
                return
            if end_t < float(self.ref_duration_s) - 1e-3:
                print(
                    f"[sim2sim] sim_duration cap reached ({self.cfg.sim_duration:.2f}s, "
                    f"ref={self.ref_duration_s:.2f}s, motion_time={u_t:.2f})",
                    flush=True,
                )
                self._run_stop_reason = "max_time"
                return
        if u_t >= float(self.ref_duration_s) - self.dt - 1e-6:
            print(
                f"[sim2sim] ref motion ended (motion_time={u_t:.2f}, "
                f"ref_duration={self.ref_duration_s:.2f}s)",
                flush=True,
            )
            self._run_stop_reason = "ref_end"
            return
        if u_t >= float(self.motion_len) - self.dt - 1e-6:
            print(f"[sim2sim] motion ended (motion_time={u_t:.2f})", flush=True)
            self._run_stop_reason = "motion_end"

    def _write_run_metadata(self) -> None:
        path = self.run_metadata_json
        if not path:
            return
        control_freq = 1.0 / float(self.dt)
        policy_freq = control_freq
        meta = {
            "full_ref": bool(self.full_ref),
            "ref_duration_s": float(self.ref_duration_s),
            "eval_duration_s": float(self.eval_duration_s),
            "motion_len_s": float(self.motion_len),
            "sim_duration": self.cfg.sim_duration,
            "t_end_s": float(self._last_motion_time),
            "stop_reason": self._run_stop_reason,
            "clip_actions_residual_rad": float(self.cfg.clip_actions),
            "clip_actions_effective_max_residual_rad": _max_residual_rad(
                float(self.cfg.clip_actions), self.action_scale
            ),
            "action_lowpass_alpha": float(self.cfg.action_filter_alpha),
            "action_filter_new_weight": float(self.cfg.action_filter_alpha),
            "action_delay": bool(self.cfg.action_delay),
            "action_delay_steps": self.cfg.action_delay_steps,
            "action_lowpass_filter": bool(self.cfg.action_lowpass_filter),
            "action_scale": [float(x) for x in self.action_scale],
            "control_freq_hz": control_freq,
            "policy_freq_hz": policy_freq,
            "action_scale_max": float(np.max(self.action_scale)),
            "csv_fps": float(self.csv_fps),
            "motion_csv": getattr(self.motion_lib, "_csv_path", None),
            "onnx": self.onnx_path,
            "turn_yaw_scale": float(self.turn_yaw_scale),
            "turn_yaw_scale_applies": bool(self._turn_yaw_scale_applies()),
            "turn_yaw_scale_motion_substrings": list(self._turn_yaw_scale_motion_substrings),
            "mimic_ref_wz_scaled": bool(self.mimic_ref_wz_scaled),
            "ref_pose_scaled": bool(self.ref_pose_scaled),
            "ref_end_hold_enabled": bool(self.ref_end_hold_enabled),
            "ref_end_hold_s": float(self.ref_end_hold_s),
            "zero_ref_vel_after_end": bool(self.zero_ref_vel_after_end),
            "reset_after_hold_if_alive": bool(self.reset_after_hold_if_alive),
            "max_episode_length_s": self.max_episode_length_s,
            "checkpoint": os.path.basename(self.onnx_path),
            "clip": float(self.cfg.clip_actions),
            "kps": [float(x) for x in self.kps_policy],
            "kds": [float(x) for x in self.kds_policy],
            "joint_order": [str(x) for x in self.cfg.dof_names],
            "root_height_offset": float(self.cfg.root_height_offset),
            "root_reset_z_offset": float(self.cfg.root_reset_z_offset),
            "reset_vel_factor": float(self.cfg.reset_vel_factor),
            "reset_ang_vel_frame": "local",
            "wrist_roll_fixed_mode": "xml_topology_fixed_omitted_from_21_actions",
            "max_episode_length": (
                int(round(float(self.max_episode_length_s) / float(self.dt)))
                if self.max_episode_length_s is not None
                else None
            ),
            "obs_inject_real_log": self.cfg.obs_inject_real_log,
            "obs_inject_upper_error": bool(self.cfg.obs_inject_upper_error),
            "obs_inject_upper_vel": bool(self.cfg.obs_inject_upper_vel),
            "obs_inject_upper_scale": float(self.cfg.obs_inject_upper_scale),
            "obs_inject_time_col": str(self.cfg.obs_inject_time_col),
            "obs_inject_time_shift_s": float(self.cfg.obs_inject_time_shift_s),
            "termination_reason": self._run_stop_reason,
            "failed": self._run_stop_reason not in _NON_FAIL_TERMINATION_REASONS,
            "first_failure_time_s": self._first_failure_time_s,
            "first_failure_reason": self._first_failure_reason or None,
            "actual_duration_s": max(
                0.0, float(self._last_motion_time) - float(self.motion_time_offset)
            ),
            "effective_horizon_s": max(
                0.0,
                float(self._motion_playback_end_time()) - float(self.motion_time_offset),
            ),
            "requested_horizon_s": getattr(self, "requested_horizon_s", None),
            "post_motion_hold_s": getattr(self, "post_motion_hold_s", None),
            "terminal_velocity_decay": str(
                getattr(self.cfg, "motion_termination_vel_decay", "zero")
            ),
            "truncated": self._run_stop_reason == "horizon_truncated",
            "limit_contract": getattr(self, "_g1_limit_contract_metadata", None),
            "canonical_abi": getattr(self, "_canonical_abi", None),
            "evaluation_contract": getattr(self, "_evaluation_contract_binding", None),
        }
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)
        print(f"[sim2sim] run metadata -> {path}", flush=True)

    def _append_inference_log(
        self,
        timestamp: float,
        base_quat_xyzw: np.ndarray,
        base_ang_vel_local: np.ndarray,
        joint_pos_policy: np.ndarray,
        joint_vel_policy: np.ndarray,
        raw_action: np.ndarray,
        applied_action: np.ndarray,
        motion_time_tensor,
        unclamped_motion_time: float,
        raw_policy_obs: Optional[np.ndarray] = None,
    ):
        if self._inference_log_writer is None:
            return
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos = self._query_ref_state(motion_time_tensor)
        rq = root_rot[0].detach().cpu().numpy().astype(np.float64)
        rp = root_pos[0].detach().cpu().numpy().astype(np.float64)
        rv = root_vel[0].detach().cpu().numpy().astype(np.float64)
        rw = root_ang_vel[0].detach().cpu().numpy().astype(np.float64)
        rdof = dof_pos[0].detach().cpu().numpy().astype(np.float64)
        bq = np.asarray(base_quat_xyzw, dtype=np.float64).reshape(-1)
        bw = np.asarray(base_ang_vel_local, dtype=np.float64).reshape(-1)
        jp = np.asarray(joint_pos_policy, dtype=np.float64).reshape(-1)
        jv = np.asarray(joint_vel_policy, dtype=np.float64).reshape(-1)
        raw_ac = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        applied_ac = np.asarray(applied_action, dtype=np.float64).reshape(-1)
        n = self.cfg.num_actions
        clip_lim = float(self.cfg.clip_actions) / self.action_scale
        clipped_ac = np.clip(raw_ac, -clip_lim, clip_lim)
        dof_pos_target = self.ref_dof_pos_res + applied_ac * self.action_scale

        base_pos = np.array(self.data.qpos[0:3], dtype=np.float64)
        base_roll, base_pitch, base_yaw = _euler_xyz_from_quat_xyzw(bq)
        base_lin_vel = np.array(self.data.qvel[0:3], dtype=np.float64)
        foot_z, foot_vz, foot_fn, foot_contact_bool, foot_vel_xy_sq = _read_feet_state(
            self.model,
            self.data,
            self._foot_body_ids,
            self._foot_geom_to_idx,
            self._ground_geom_ids,
            contact_force_threshold=float(CP_RMG_V1["feet_contact_force_threshold"]),
        )
        ref_contact_prob = _estimate_ref_feet_contact_prob(rdof[:n], self.cfg.dof_names)
        reward_terms = _compute_feet_reward_terms(
            foot_z,
            foot_fn,
            foot_vel_xy_sq,
            ref_contact_prob,
            ankle_2_ground=float(CP_RMG_V1["ankle_2_ground"]),
            feet_clearance_simple_target=float(CP_RMG_V1["feet_clearance_simple_target"]),
            feet_clearance_simple_tolerance=float(CP_RMG_V1["feet_clearance_simple_tolerance"]),
            feet_use_force_contact=bool(CP_RMG_V1["feet_use_force_contact"]),
            feet_contact_force_threshold=float(CP_RMG_V1["feet_contact_force_threshold"]),
            feet_contact_force_k=float(CP_RMG_V1["feet_contact_force_k"]),
            feet_contact_conf_gamma=float(CP_RMG_V1["feet_contact_conf_gamma"]),
            feet_contact_alpha=float(CP_RMG_V1["feet_contact_alpha"]),
            feet_clearance_alpha=float(CP_RMG_V1["feet_clearance_alpha"]),
            feet_min_clearance=float(CP_RMG_V1["feet_min_clearance"]),
            feet_scuff_lambda=float(CP_RMG_V1["feet_scuff_lambda"]),
            feet_scuff_height_k=float(CP_RMG_V1["feet_scuff_height_k"]),
        )
        episode_done, termination_reason = self._episode_termination_flags(unclamped_motion_time)

        ref_swing = np.clip((0.999 - ref_contact_prob) / (0.999 - 0.001), 0.0, 1.0)
        stance_mask = 1.0 - ref_swing
        feet_stance_slip = float(np.sum(stance_mask * foot_vel_xy_sq))
        feet_contact_match = float(reward_terms["reward/feet_contact_matching"])
        touchdown_peak_force = 0.0
        touchdown_vertical_velocity = 0.0
        for fi in range(2):
            rising = bool(foot_contact_bool[fi]) and not self._prev_foot_contact_bool[fi]
            if rising:
                touchdown_peak_force = max(touchdown_peak_force, float(foot_fn[fi]))
                touchdown_vertical_velocity = max(
                    touchdown_vertical_velocity, abs(float(foot_vz[fi]))
                )
        self._prev_foot_contact_bool = [bool(foot_contact_bool[0]), bool(foot_contact_bool[1])]

        row = [
            float(timestamp),
            float(bq[3]),
            float(bq[0]),
            float(bq[1]),
            float(bq[2]),
            float(bw[0]),
            float(bw[1]),
            float(bw[2]),
        ]
        row.extend(float(x) for x in jp[:n])
        row.extend(float(x) for x in jv[:n])
        row.extend(float(x) for x in applied_ac[:n])
        row.extend(float(x) for x in raw_ac[:n])
        row.extend(float(x) for x in clipped_ac[:n])
        row.extend(float(x) for x in applied_ac[:n])
        row.extend(float(x) for x in dof_pos_target[:n])
        row.extend(
            [
                float(rp[2]),
                float(rq[3]),
                float(rq[0]),
                float(rq[1]),
                float(rq[2]),
                float(rv[0]),
                float(rv[1]),
                float(rv[2]),
                float(rw[0]),
                float(rw[1]),
                float(rw[2]),
            ]
        )
        row.extend(float(x) for x in rdof[:n])
        row.extend(
            [
                float(base_roll),
                float(base_pitch),
                float(base_yaw),
                float(bw[0]),
                float(bw[1]),
                float(bw[2]),
                float(foot_z[0]),
                float(foot_z[1]),
                float(foot_vz[0]),
                float(foot_vz[1]),
                float(foot_fn[0]),
                float(foot_fn[1]),
                int(bool(foot_contact_bool[0])),
                int(bool(foot_contact_bool[1])),
                float(touchdown_peak_force),
                float(touchdown_vertical_velocity),
                feet_contact_match,
                feet_stance_slip,
                float(base_lin_vel[0]),
                float(base_lin_vel[1]),
                float(base_lin_vel[2]),
                float(base_pos[0]),
                float(base_pos[1]),
                float(base_pos[2]),
                int(episode_done),
                termination_reason,
                float(reward_terms["reward/feet_clearance_simple"]),
                float(reward_terms["reward/feet_contact_matching"]),
                float(reward_terms["reward/feet_swing_clearance"]),
                float(reward_terms["reward/feet_scuffing"]),
            ]
        )
        if self.cfg.inference_log_policy_obs or self.cfg.inference_log_full_policy_input:
            if raw_policy_obs is None:
                raw_policy_obs = np.zeros((N_POLICY_OBS,), dtype=np.float64)
            else:
                raw_policy_obs = np.asarray(raw_policy_obs, dtype=np.float64).reshape(-1)
            if raw_policy_obs.size < N_OBS_SINGLE:
                raise ValueError(
                    f"raw_policy_obs size {raw_policy_obs.size} < N_OBS_SINGLE {N_OBS_SINGLE}"
                )
            if self.cfg.inference_log_policy_obs:
                row.extend(float(raw_policy_obs[i]) for i in range(N_OBS_SINGLE))
            if self.cfg.inference_log_full_policy_input:
                if raw_policy_obs.size < N_POLICY_OBS:
                    raise ValueError(
                        f"raw_policy_obs size {raw_policy_obs.size} < N_POLICY_OBS {N_POLICY_OBS}"
                    )
                row.extend(float(raw_policy_obs[i]) for i in range(N_POLICY_OBS))
        self._inference_log_writer.writerow(row)

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
        self._act_qposadr = np.asarray([int(self.model.jnt_qposadr[j]) for j in act_joint_ids], dtype=np.int32)
        self._act_qveladr = np.asarray([int(self.model.jnt_dofadr[j]) for j in act_joint_ids], dtype=np.int32)
        self.kps_act = self.kps_policy[self._act_to_policy]
        self.kds_act = self.kds_policy[self._act_to_policy]
        self._build_motor_limits()
        self.waist_yaw_policy_idx = self.cfg.dof_names.index("waist_yaw_joint")
        self.left_ankle_pitch_policy_idx = self.cfg.dof_names.index("left_leg_ankle_pitch_joint")
        self.right_ankle_pitch_policy_idx = self.cfg.dof_names.index("right_leg_ankle_pitch_joint")
        upper_tokens = ("waist", "shoulder", "elbow", "wrist", "arm")
        upper_body_idx = [i for i, n in enumerate(self.cfg.dof_names) if any(tok in n for tok in upper_tokens)]
        self._upper_body_dof_idx = np.asarray(upper_body_idx, dtype=np.int64)
        self._upper_key_body_dim = 3 * len(self.cfg.upper_key_bodies)
        self._load_obs_injection_profile()

    def _load_obs_injection_profile(self) -> None:
        self._obs_inject_profile = None
        log_path = self.cfg.obs_inject_real_log
        if not log_path:
            return

        path = os.path.abspath(log_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"obs injection log not found: {path}")

        df = pd.read_csv(path)
        num_actions = self.cfg.num_actions
        time_col_pref = self.cfg.obs_inject_time_col
        if time_col_pref == "auto":
            time_col = None
            for candidate in ("ref_time_s", "global_time_s", "timestamp"):
                if candidate in df.columns:
                    time_col = candidate
                    break
            if time_col is None:
                raise ValueError(
                    "obs injection log missing time column; tried: ref_time_s, global_time_s, timestamp"
                )
        else:
            time_col = time_col_pref
            if time_col not in df.columns:
                raise ValueError(f"obs injection log missing time column: {time_col}")

        pos_cols = [f"dof_pos_minus_ref_{i}" for i in range(num_actions)]
        vel_cols = [f"dof_vel_{i}" for i in range(num_actions)]
        missing = [c for c in pos_cols + vel_cols if c not in df.columns]
        if missing:
            raise ValueError(f"obs injection log missing columns: {missing}")

        sub_df = df[[time_col] + pos_cols + vel_cols].copy()
        sub_df = sub_df.rename(columns={time_col: "time"})
        sub_df = sub_df.sort_values("time").drop_duplicates(subset="time", keep="last")
        time_arr = sub_df["time"].to_numpy(dtype=np.float64)
        pos_minus_ref = sub_df[pos_cols].to_numpy(dtype=np.float32)
        vel = sub_df[vel_cols].to_numpy(dtype=np.float32)
        if time_arr.size < 2:
            raise ValueError(f"obs injection log needs at least 2 unique time rows, got {time_arr.size}")

        self._obs_inject_profile = {
            "time": time_arr,
            "pos_minus_ref": pos_minus_ref,
            "vel": vel,
            "time_col": time_col,
        }
        print(
            f"[sim2sim] obs injection loaded: path={path}, rows={len(time_arr)}, "
            f"time_col={time_col}, range=[{time_arr[0]:.4f}, {time_arr[-1]:.4f}], "
            f"upper_scale={self.cfg.obs_inject_upper_scale}",
            flush=True,
        )

    def _interp_obs_injection_profile(self, motion_time: float) -> tuple[np.ndarray, np.ndarray] | None:
        profile = self._obs_inject_profile
        if profile is None:
            return None
        log_t = float(motion_time) + float(self.cfg.obs_inject_time_shift_s)
        t = profile["time"]
        if log_t < t[0] or log_t > t[-1]:
            return None
        pos_minus_ref = np.empty((self.cfg.num_actions,), dtype=np.float32)
        vel = np.empty((self.cfg.num_actions,), dtype=np.float32)
        for i in range(self.cfg.num_actions):
            pos_minus_ref[i] = np.interp(log_t, t, profile["pos_minus_ref"][:, i])
            vel[i] = np.interp(log_t, t, profile["vel"][:, i])
        return pos_minus_ref, vel

    @staticmethod
    def _apply_policy_action_ablations(cfg: RMGConfig, action: np.ndarray) -> np.ndarray:
        """兼容 shape (num_actions,) 或 (1, num_actions)；返回施加消融后的 residuals。"""
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if cfg.zero_policy_action:
            a[:] = 0.0
            return a
        if cfg.zero_hip_roll_and_ankle_roll_action:
            for i in HIP_ROLL_ACTION_INDICES:
                if i < a.size:
                    a[i] = 0.0
            for i in ANKLE_ROLL_ACTION_INDICES:
                if i < a.size:
                    a[i] = 0.0
            return a
        if cfg.zero_ankle_roll_action:
            for i in ANKLE_ROLL_ACTION_INDICES:
                if i < a.size:
                    a[i] = 0.0
        return a

    def _build_motor_limits(self):
        effort_policy, vel_policy, pos_limits_policy = self._load_urdf_joint_limits()
        if effort_policy is None:
            ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
            effort_act = np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1]))
            if np.all(effort_act < 1e-6):
                effort_act = np.full((self.cfg.num_actions,), np.inf, dtype=np.float32)
            vel_act = np.full((self.cfg.num_actions,), np.inf, dtype=np.float32)
        else:
            effort_act = self._policy_to_act_vec(effort_policy)
            vel_act = self._policy_to_act_vec(vel_policy)
        self.torque_limits = effort_act * float(self.cfg.torque_safety_limit)
        self.motor_tau_stall = effort_act * float(self.cfg.urdf_limit_discount)
        self.motor_omega_nl = vel_act * float(self.cfg.urdf_limit_discount)
        self.dof_pos_limits_policy = pos_limits_policy

    def _load_urdf_joint_limits(self):
        urdf_path = os.path.abspath(self.cfg.urdf_path)
        if not os.path.exists(urdf_path):
            return None, None, None
        root = ET.parse(urdf_path).getroot()
        limits_by_joint = {}
        for joint in root.iter("joint"):
            name = joint.attrib.get("name")
            limit = joint.find("limit")
            if not name or limit is None:
                continue
            effort = limit.attrib.get("effort")
            velocity = limit.attrib.get("velocity")
            lower = limit.attrib.get("lower")
            upper = limit.attrib.get("upper")
            if effort is None or velocity is None or lower is None or upper is None:
                continue
            limits_by_joint[name] = (
                float(effort),
                float(velocity),
                float(lower),
                float(upper),
            )

        missing = [name for name in self.cfg.dof_names if name not in limits_by_joint]
        if missing:
            raise ValueError(f"URDF missing motor limits for joints: {missing}")
        effort = np.asarray([limits_by_joint[name][0] for name in self.cfg.dof_names], dtype=np.float32)
        velocity = np.asarray([limits_by_joint[name][1] for name in self.cfg.dof_names], dtype=np.float32)
        pos_limits = np.asarray(
            [[limits_by_joint[name][2], limits_by_joint[name][3]] for name in self.cfg.dof_names],
            dtype=np.float32,
        )
        return effort, velocity, pos_limits

    def _load_urdf_motor_limits(self):
        effort, velocity, _ = self._load_urdf_joint_limits()
        return effort, velocity

    def _compute_balance_safe_pd_targets(self, action_cmd: np.ndarray) -> np.ndarray:
        return compute_balance_safe_pd_targets(
            self.ref_dof_pos_res,
            action_cmd,
            self.action_scale,
            enable_target_offset_clip=bool(self.cfg.enable_target_offset_clip),
            target_offset_clip=self._target_offset_clip,
            enable_target_joint_limit_clamp=bool(self.cfg.enable_target_joint_limit_clamp),
            dof_pos_limits=self.dof_pos_limits_policy,
            target_joint_limit_margin=float(self.cfg.target_joint_limit_margin),
        )

    def _policy_action_to_dof_pos_target(self, action_cmd: np.ndarray) -> np.ndarray:
        if self.cfg.g1_balance_safe_enable:
            target = self._compute_balance_safe_pd_targets(action_cmd)
        else:
            target = self.ref_dof_pos_res + action_cmd * self.action_scale
        target = target.astype(np.float32, copy=False)
        target[self.left_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_left)
        target[self.right_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_right)
        return target

    def _build_body_mappings(self):
        body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        name_to_id = {n: i for i, n in enumerate(body_names)}
        missing = [n for n in self.cfg.upper_key_bodies if n not in name_to_id]
        if missing:
            raise ValueError(f"MuJoCo model missing upper key bodies: {missing}")

    def _configure_model(self):
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            self.model.geom_friction[floor_id][0] = self.cfg.friction_coeffs

    def _act_to_policy_vec(self, vec_act):
        return vec_act[self._policy_to_act]

    def _policy_to_act_vec(self, vec_policy):
        return vec_policy[self._act_to_policy]

    def _get_dof_state_act(self):
        return self.data.qpos[self._act_qposadr].astype(np.float32, copy=False), self.data.qvel[self._act_qveladr].astype(np.float32, copy=False)

    def _get_base_quat_xyzw(self):
        q = self.data.qpos
        return np.array([q[4], q[5], q[6], q[3]], dtype=np.float32)

    def _get_base_ang_vel_local(self, base_quat):
        """与 Isaac `base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])` 一致。"""
        dq = self.data.qvel.astype(np.float32)
        world_ang_vel = torch.tensor(dq[3:6], dtype=torch.float32, device=self.device).unsqueeze(0)
        return quat_rotate_inverse(base_quat, world_ang_vel)

    def _reset_yaw_align_state(self) -> None:
        self._yaw_align_offset = 0.0
        self._yaw_align_capture_pending = bool(self.cfg.yaw_align_to_ref)

    def _get_base_quat_for_policy(self) -> np.ndarray:
        """策略 obs 用四元数：可选 yaw 对齐 ref[0]。"""
        q_raw = self._get_base_quat_xyzw().astype(np.float64)
        self._robot_yaw_raw = _yaw_from_quat_xyzw(q_raw)

        if self.cfg.yaw_align_to_ref and self._yaw_align_capture_pending:
            ref_time = torch.tensor([self.motion_time_offset], device=self.device, dtype=torch.float32)
            _, ref_rot, _, _, _, _, _ = self._calc_motion_frame(self.motion_ids, ref_time)
            ref_q = ref_rot[0].detach().cpu().numpy().astype(np.float64)
            ref_yaw = _yaw_from_quat_xyzw(ref_q)
            delta = _wrap_to_pi(ref_yaw - self._robot_yaw_raw)
            self._yaw_align_offset = delta
            self._yaw_align_capture_pending = False
            print(
                f"[yaw_align] imu={np.degrees(self._robot_yaw_raw):.1f}° "
                f"ref0={np.degrees(ref_yaw):.1f}° offset={np.degrees(delta):.1f}° "
                f"full_proprio={self.cfg.yaw_align_full_proprio}",
                flush=True,
            )

        if self.cfg.yaw_align_to_ref:
            return _apply_yaw_align_xyzw(q_raw, self._yaw_align_offset)
        return q_raw.astype(np.float32)

    def _obs_orientation_tensors(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """返回 (quat_gravity, quat_ang_vel, q_raw, q_policy) 供 proprio 构造。"""
        q_raw = self._get_base_quat_xyzw()
        q_raw_t = torch.tensor(q_raw, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not self.cfg.yaw_align_to_ref:
            return q_raw_t, q_raw_t, q_raw, q_raw
        q_policy = self._get_base_quat_for_policy()
        q_policy_t = torch.tensor(q_policy, dtype=torch.float32, device=self.device).unsqueeze(0)
        quat_ang_vel = q_raw_t if not self.cfg.yaw_align_full_proprio else q_policy_t
        return q_policy_t, quat_ang_vel, q_raw, q_policy

    def _turn_yaw_scale_applies(self) -> bool:
        if abs(float(self.turn_yaw_scale) - 1.0) < 1e-6:
            return False
        name = os.path.basename(self._motion_csv_path)
        return any(s in name for s in self._turn_yaw_scale_motion_substrings)

    def _apply_turn_yaw_scale_world(self, root_ang_vel: torch.Tensor) -> torch.Tensor:
        """Match Isaac _apply_turn_yaw_curriculum_to_ref: scale world-frame ref yaw rate only."""
        if not self._turn_yaw_scale_applies():
            return root_ang_vel
        out = root_ang_vel.clone()
        out[..., 2] = out[..., 2] * float(self.turn_yaw_scale)
        return out

    def _clamp_motion_query_time(self, motion_time: torch.Tensor) -> torch.Tensor:
        max_t = torch.tensor([max(float(self.motion_len) - 1e-4, 0.0)], device=self.device, dtype=torch.float32)
        return torch.minimum(motion_time, max_t)

    def _motion_time_past_ref_end(self, unclamped_motion_time: float) -> float:
        return max(0.0, float(unclamped_motion_time) - float(self.ref_duration_s))

    def _calc_motion_frame(self, motion_ids, motion_times):
        query_t = self._clamp_motion_query_time(motion_times)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos = (
            self.motion_lib.calc_motion_frame(motion_ids, query_t)
        )
        root_ang_vel = self._apply_turn_yaw_scale_world(root_ang_vel)
        if self.ref_end_hold_enabled and self.zero_ref_vel_after_end:
            raw_t = motion_times.reshape(-1)
            past = raw_t > (float(self.ref_duration_s) + 1e-6)
            if bool(past.any().item()):
                decay = str(
                    getattr(self.cfg, "motion_termination_vel_decay", "zero")
                ).lower()
                if decay == "cosine":
                    hold_s = max(
                        float(
                            getattr(
                                self.cfg,
                                "motion_termination_hold_time",
                                self.ref_end_hold_s,
                            )
                        ),
                        1.0e-6,
                    )
                    alpha = torch.clamp(
                        (raw_t - float(self.ref_duration_s)) / hold_s,
                        min=0.0,
                        max=1.0,
                    )
                    vel_scale = 0.5 * (1.0 + torch.cos(alpha * np.pi))
                elif decay == "linear":
                    hold_s = max(float(self.ref_end_hold_s), 1.0e-6)
                    vel_scale = torch.clamp(
                        1.0
                        - (raw_t - float(self.ref_duration_s)) / hold_s,
                        min=0.0,
                        max=1.0,
                    )
                else:
                    vel_scale = torch.where(
                        past,
                        torch.zeros_like(raw_t),
                        torch.ones_like(raw_t),
                    )
                root_vel = root_vel * vel_scale.view(-1, 1)
                root_ang_vel = root_ang_vel * vel_scale.view(-1, 1)
                dof_vel = dof_vel * vel_scale.view(-1, 1)
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos

    def _query_ref_state(self, motion_time):
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _ = self._calc_motion_frame(
            self.motion_ids, motion_time
        )
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos

    def _build_mimic_obs(self, motion_time):
        """mimic 航向槽仅用参考轨迹未来转向意图，不使用机器人实测 yaw。"""
        query_t = motion_time + float(self.cfg.mimic_tar_first_step) * self.dt
        max_t = torch.tensor([max(float(self.motion_len) - 0.01, 0.0)], device=self.device, dtype=torch.float32)
        query_t = torch.minimum(query_t, max_t)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _ = self._calc_motion_frame(
            self.motion_ids, query_t
        )
        t_future = query_t + float(self.cfg.ref_turn_intent_horizon_s)
        t_future = torch.minimum(t_future, max_t)
        _, root_rot_future, _, _, _, _, _ = self._calc_motion_frame(self.motion_ids, t_future)
        heading_sin, heading_cos = ref_turn_intent_heading_sin_cos(root_rot, root_rot_future)
        roll, pitch, _ = euler_from_quaternion(root_rot)
        ref_vel_local = quat_rotate_inverse(root_rot, root_vel)
        ref_ang_local = quat_rotate_inverse(root_rot, root_ang_vel)
        self.ref_dof_pos_res = dof_pos[0].detach().cpu().numpy().astype(np.float32)
        return torch.cat(
            (
                root_pos[:, 2:3],
                roll.unsqueeze(-1),
                pitch.unsqueeze(-1),
                heading_sin,
                heading_cos,
                ref_vel_local,
                ref_ang_local[:, 2:3],
                dof_pos,
            ),
            dim=-1,
        )

    def _build_actor_obs(self, motion_time):
        _, ref_root_rot, _, ref_root_ang_vel, ref_dof_pos_cur = self._query_ref_state(motion_time)
        quat_gravity, quat_ang_vel, q_raw, q_policy = self._obs_orientation_tensors()
        base_ang_vel = self._get_base_ang_vel_local(quat_ang_vel)
        projected_gravity = quat_rotate_inverse(quat_gravity, self.gravity_vec)
        dof_pos_act, dof_vel_act = self._get_dof_state_act()
        dof_pos_t = torch.tensor(self._act_to_policy_vec(dof_pos_act), dtype=torch.float32, device=self.device).unsqueeze(0)
        dof_vel_t = torch.tensor(self._act_to_policy_vec(dof_vel_act), dtype=torch.float32, device=self.device).unsqueeze(0)
        dof_pos_obs_t = dof_pos_t
        dof_vel_obs_t = dof_vel_t
        if self.cfg.obs_inject_real_log and (
            self.cfg.obs_inject_upper_error or self.cfg.obs_inject_upper_vel
        ):
            profile = self._interp_obs_injection_profile(float(motion_time.reshape(-1)[0].item()))
            if profile is not None:
                pos_minus_ref_np, vel_np = profile
                idx = self._upper_body_dof_idx
                dof_pos_obs_t = dof_pos_t.clone()
                dof_vel_obs_t = dof_vel_t.clone()
                scale = float(self.cfg.obs_inject_upper_scale)
                if self.cfg.obs_inject_upper_error:
                    pos_err = torch.tensor(
                        pos_minus_ref_np[idx] * scale, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    dof_pos_obs_t[:, idx] = ref_dof_pos_cur[:, idx] + pos_err
                if self.cfg.obs_inject_upper_vel:
                    vel_obs = torch.tensor(
                        vel_np[idx] * scale, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    dof_vel_obs_t[:, idx] = vel_obs
        ref_root_ang_vel_local = quat_rotate_inverse(ref_root_rot, ref_root_ang_vel)
        root_ang_vel_err_yaw = ref_root_ang_vel_local[:, 2:3] - base_ang_vel[:, 2:3]
        if self.cfg.yaw_rate_err_soft_clip:
            root_ang_vel_err_yaw = _soft_clamp_obs(
                root_ang_vel_err_yaw, self.cfg.yaw_rate_err_clip
            )
        waist_pos = (dof_pos_obs_t[:, self.waist_yaw_policy_idx:self.waist_yaw_policy_idx + 1] - self.default_dof_pos_t[:, self.waist_yaw_policy_idx:self.waist_yaw_policy_idx + 1]) * self.cfg.obs_dof_pos_scale
        waist_vel = dof_vel_obs_t[:, self.waist_yaw_policy_idx:self.waist_yaw_policy_idx + 1] * self.cfg.obs_dof_vel_scale

        upper_joint_err = (ref_dof_pos_cur - dof_pos_obs_t)[:, self._upper_body_dof_idx]
        n_u = upper_joint_err.shape[1]
        upper_dim = self._upper_key_body_dim
        rep = (upper_dim + n_u - 1) // n_u if n_u > 0 else 1
        if n_u == 0:
            upper_key_body_err_rmg = torch.zeros((1, upper_dim), dtype=torch.float32, device=self.device)
        else:
            upper_key_body_err_rmg = upper_joint_err.repeat(1, rep)[:, :upper_dim]
        tracking_err_rmg = torch.cat((root_ang_vel_err_yaw, upper_key_body_err_rmg), dim=-1)
        arm_idx = self._upper_body_dof_idx[self._upper_body_dof_idx != self.waist_yaw_policy_idx]
        if arm_idx.size == 0:
            hand_proxy = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        else:
            arm_pos = (dof_pos_obs_t[:, arm_idx] - self.default_dof_pos_t[:, arm_idx]) * self.cfg.obs_dof_pos_scale
            rep_h = (6 + arm_pos.shape[1] - 1) // arm_pos.shape[1]
            hand_proxy = arm_pos.repeat(1, rep_h)[:, :6]
        upper_body_rmg = torch.cat((waist_pos, waist_vel, base_ang_vel[:, :2], hand_proxy), dim=-1)
        proprio_current = torch.cat(
            (
                base_ang_vel * self.cfg.obs_ang_vel_scale,
                projected_gravity,
                (dof_pos_obs_t - self.default_dof_pos_t) * self.cfg.obs_dof_pos_scale,
                dof_vel_obs_t * self.cfg.obs_dof_vel_scale,
                self.action_tensor,
                tracking_err_rmg,
                upper_body_rmg,
            ),
            dim=-1,
        )
        if proprio_current.shape[-1] != N_PROPRIO_OBS:
            raise ValueError(f"Expected proprio dim {N_PROPRIO_OBS}, got {proprio_current.shape[-1]}")

        for idx in self.cfg.ankle_idx:
            proprio_current[:, 6 + self.cfg.num_actions + idx] = 0.0

        mimic_obs = self._build_mimic_obs(motion_time)
        if mimic_obs.shape[-1] != N_MIMIC_OBS:
            raise ValueError(f"Expected mimic dim {N_MIMIC_OBS}, got {mimic_obs.shape[-1]}")
        student_obs = torch.cat((mimic_obs, proprio_current), dim=-1)
        if student_obs.shape[-1] != N_OBS_SINGLE:
            raise ValueError(f"Expected student_obs dim {N_OBS_SINGLE}, got {student_obs.shape[-1]}")
        # 与 compute_observations: cat([current, history_flat]) 一致
        policy_obs = torch.cat((student_obs, self.obs_history_buf.reshape(1, -1)), dim=-1)
        if policy_obs.shape[-1] != N_POLICY_OBS:
            raise ValueError(f"Expected policy obs dim {N_POLICY_OBS}, got {policy_obs.shape[-1]}")
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([student_obs] * self.cfg.history_len, dim=1),
            torch.cat((self.obs_history_buf[:, 1:], student_obs.unsqueeze(1)), dim=1),
        )
        return policy_obs

    @staticmethod
    def _clip_raw_obs_for_onnx(raw_obs: torch.Tensor, clip_observations: float) -> torch.Tensor:
        """与 env.step 返回前 obs_buf clip 一致；归一化在 ONNX 图内完成。"""
        return torch.clip(raw_obs, -clip_observations, clip_observations)

    def _infer_onnx_actions(self, raw_obs_clipped: torch.Tensor) -> np.ndarray:
        """ONNX 输入为 raw+clip 的 obs；输出为 raw action，不做 normalize。"""
        obs_np = raw_obs_clipped.detach().cpu().numpy().astype(np.float32)
        return self.policy.run([self.policy_output_name], {self.policy_input_name: obs_np})[0].reshape(-1).astype(np.float32)

    def _build_diag_track_ctx(
        self,
        motion_time_tensor,
        unclamped_motion_time: float,
        target_dof_pos_policy: np.ndarray,
        tau_raw_act: np.ndarray,
        dof_pos_policy: np.ndarray,
        policy_action_effective: np.ndarray,
    ) -> dict:
        """诊断专用；不参与 policy。含 world/local 速度与踝/髋 roll 分项。"""
        root_pos, root_rot, root_vel, root_ang_vel, _ = self._query_ref_state(motion_time_tensor)
        ref_pos = root_pos[0].detach().cpu().numpy().astype(np.float64)
        ref_q = root_rot[0].detach().cpu().numpy().astype(np.float64)
        ref_vel_w = root_vel[0].detach().cpu().numpy().astype(np.float64)
        ref_ang_w = root_ang_vel[0].detach().cpu().numpy().astype(np.float64)
        ref_vel_local = _quat_rotate_inverse_np(ref_q, ref_vel_w)
        ref_ang_local = _quat_rotate_inverse_np(ref_q, ref_ang_w)
        base_quat = self._get_base_quat_xyzw().astype(np.float64)
        world_lin = np.array(self.data.qvel[0:3], dtype=np.float64)
        world_ang = np.array(self.data.qvel[3:6], dtype=np.float64)
        base_vel_local = _quat_rotate_inverse_np(base_quat, world_lin)
        base_ang_local = _quat_rotate_inverse_np(base_quat, world_ang)
        br, bp, by = _euler_xyz_from_quat_xyzw(base_quat)
        base_rpy = np.array([br, bp, by], dtype=np.float64)
        vel_err_xy = ref_vel_local[:2] - base_vel_local[:2]
        base_xy = np.array(self.data.qpos[0:2], dtype=np.float64)
        root_xy_err = ref_pos[:2] - base_xy
        tgt = np.asarray(target_dof_pos_policy, dtype=np.float64).reshape(-1)
        actp = np.asarray(dof_pos_policy, dtype=np.float64).reshape(-1)
        ai = self.cfg.ankle_idx
        ankle_tgt = np.array([tgt[int(i)] for i in ai if int(i) < tgt.size], dtype=np.float64)
        ankle_act = np.array([actp[int(i)] for i in ai if int(i) < actp.size], dtype=np.float64)
        mt = float(motion_time_tensor.reshape(-1)[0].item()) if motion_time_tensor.numel() else float("nan")
        motion_len = float(self.motion_len)
        motion_phase_valid = bool(unclamped_motion_time < motion_len - 0.01)
        peff = np.asarray(policy_action_effective, dtype=np.float64).reshape(-1)
        ar_idx = ANKLE_ROLL_ACTION_INDICES
        hr_idx = HIP_ROLL_ACTION_INDICES
        ankle_roll_tgt = np.array([tgt[i] for i in ar_idx if i < tgt.size], dtype=np.float64)
        ankle_roll_act_pos = np.array([actp[i] for i in ar_idx if i < actp.size], dtype=np.float64)
        hip_roll_tgt = np.array([tgt[i] for i in hr_idx if i < tgt.size], dtype=np.float64)
        hip_roll_act_pos = np.array([actp[i] for i in hr_idx if i < actp.size], dtype=np.float64)
        ankle_roll_action = np.array([peff[i] for i in ar_idx if i < peff.size], dtype=np.float64)
        hip_roll_action = np.array([peff[i] for i in hr_idx if i < peff.size], dtype=np.float64)
        yaw_drift = float("nan")
        if self._init_yaw_diag is not None:
            yaw_drift = float(by - self._init_yaw_diag)
        return {
            "motion_time": mt,
            "motion_time_unclamped": float(unclamped_motion_time),
            "motion_length": motion_len,
            "motion_phase_valid": 1.0 if motion_phase_valid else 0.0,
            "ref_root_pos": ref_pos,
            "ref_root_quat_xyzw": ref_q,
            "ref_vel_world_xy": ref_vel_w[:2].copy(),
            "ref_vel_local": ref_vel_local,
            "ref_ang_vel_local": ref_ang_local,
            "base_quat_xyzw": base_quat,
            "base_rpy": base_rpy,
            "base_vel_world_xy": world_lin[:2].copy(),
            "base_lin_vel_local": base_vel_local,
            "base_ang_vel_local": base_ang_local,
            "vel_err_xy": vel_err_xy,
            "root_xy_err_world": root_xy_err,
            "base_xy_world": base_xy.copy(),
            "yaw_drift": yaw_drift,
            "dof_pos_policy": actp,
            "ankle_tgt": ankle_tgt,
            "ankle_act": ankle_act,
            "ankle_err": ankle_tgt - ankle_act,
            "ankle_roll_target": ankle_roll_tgt,
            "ankle_roll_dof_pos": ankle_roll_act_pos,
            "ankle_roll_action": ankle_roll_action,
            "hip_roll_target": hip_roll_tgt,
            "hip_roll_dof_pos": hip_roll_act_pos,
            "hip_roll_action": hip_roll_action,
            "tau_raw": np.asarray(tau_raw_act, dtype=np.float64).reshape(-1),
        }

    def _motion_time_for_ref_row(self, ref_row: int, csv_fps: float) -> float:
        n_frames = int(self.motion_lib._motion_num_frames[0].item())
        row = int(np.clip(ref_row, 0, max(0, n_frames - 1)))
        return float(row) / float(csv_fps)

    def _set_obs_sanity_physics_state(
        self,
        case: dict,
        ref_motion_time: float,
        *,
        zero_vel: bool,
        use_real_dof: bool,
        use_real_last_action: bool,
    ) -> None:
        motion_time_t = torch.tensor([ref_motion_time], device=self.device, dtype=torch.float32)
        root_pos, root_rot, root_vel, root_ang_vel, ref_dof_pos, ref_dof_vel, _ = self._calc_motion_frame(
            self.motion_ids, motion_time_t
        )
        roll = np.deg2rad(float(case["roll_deg"]))
        pitch = np.deg2rad(float(case["pitch_deg"]))
        yaw = np.deg2rad(float(case["yaw_deg"]))
        base_quat_xyzw = _quat_xyzw_from_rpy_rad(roll, pitch, yaw)

        dof_pos_policy = ref_dof_pos[0].detach().cpu().numpy().astype(np.float32)
        dof_vel_policy = np.zeros((self.cfg.num_actions,), dtype=np.float32)
        if use_real_dof and case.get("dof_pos") is not None:
            dof_pos_policy = np.asarray(case["dof_pos"], dtype=np.float32).reshape(-1)
        if use_real_dof and case.get("dof_vel") is not None:
            dof_vel_policy = np.asarray(case["dof_vel"], dtype=np.float32).reshape(-1)

        self.data.qpos[self._act_qposadr] = self._policy_to_act_vec(dof_pos_policy)
        self.data.qvel[self._act_qveladr] = self._policy_to_act_vec(dof_vel_policy)
        rp = root_pos[0].detach().cpu().numpy().astype(np.float32)
        self.data.qpos[:3] = rp
        self.data.qpos[2] += float(self.cfg.root_height_offset)
        self.data.qpos[3] = float(base_quat_xyzw[3])
        self.data.qpos[4:7] = base_quat_xyzw[:3]

        if zero_vel:
            self.data.qvel[:] = 0.0
        else:
            self.data.qvel[:3] = 0.0
            self.data.qvel[self._act_qveladr] = self._policy_to_act_vec(dof_vel_policy)
            if case.get("base_ang_vel_x") is not None:
                local_ang = np.array(
                    [
                        float(case["base_ang_vel_x"]),
                        float(case["base_ang_vel_y"]),
                        float(case["base_ang_vel_z"]),
                    ],
                    dtype=np.float64,
                )
                world_ang = _quat_rotate_forward_np(base_quat_xyzw.astype(np.float64), local_ang)
                self.data.qvel[3:6] = world_ang.astype(np.float32)

        last_action = np.zeros((self.cfg.num_actions,), dtype=np.float32)
        if use_real_last_action:
            for key in ("actually_sent_action", "applied_action", "raw_action"):
                if case.get(key) is not None:
                    last_action = np.asarray(case[key], dtype=np.float32).reshape(-1)
                    break
        self.action_tensor[:] = torch.tensor(last_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.filtered_actions[:] = 0.0
        self.action_history_buf[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _build_student_obs_sanity(self, motion_time: float) -> tuple[torch.Tensor, dict]:
        """构造 125 维 student_obs 及诊断中间量（不更新 obs history）。"""
        _, ref_root_rot, _, ref_root_ang_vel, ref_dof_pos_cur = self._query_ref_state(
            torch.tensor([motion_time], device=self.device, dtype=torch.float32)
        )
        quat_gravity, quat_ang_vel, q_raw, q_policy = self._obs_orientation_tensors()
        base_ang_vel = self._get_base_ang_vel_local(quat_ang_vel)
        projected_gravity = quat_rotate_inverse(quat_gravity, self.gravity_vec)
        dof_pos_act, dof_vel_act = self._get_dof_state_act()
        dof_pos_t = torch.tensor(self._act_to_policy_vec(dof_pos_act), dtype=torch.float32, device=self.device).unsqueeze(0)
        dof_vel_t = torch.tensor(self._act_to_policy_vec(dof_vel_act), dtype=torch.float32, device=self.device).unsqueeze(0)
        ref_root_ang_vel_local = quat_rotate_inverse(ref_root_rot, ref_root_ang_vel)
        root_ang_vel_err_yaw = ref_root_ang_vel_local[:, 2:3] - base_ang_vel[:, 2:3]
        if self.cfg.yaw_rate_err_soft_clip:
            root_ang_vel_err_yaw = _soft_clamp_obs(
                root_ang_vel_err_yaw, self.cfg.yaw_rate_err_clip
            )
        waist_pos = (dof_pos_t[:, self.waist_yaw_policy_idx:self.waist_yaw_policy_idx + 1] - self.default_dof_pos_t[:, self.waist_yaw_policy_idx:self.waist_yaw_policy_idx + 1]) * self.cfg.obs_dof_pos_scale
        waist_vel = dof_vel_t[:, self.waist_yaw_policy_idx:self.waist_yaw_policy_idx + 1] * self.cfg.obs_dof_vel_scale

        upper_joint_err = (ref_dof_pos_cur - dof_pos_t)[:, self._upper_body_dof_idx]
        n_u = upper_joint_err.shape[1]
        upper_dim = self._upper_key_body_dim
        rep = (upper_dim + n_u - 1) // n_u if n_u > 0 else 1
        if n_u == 0:
            upper_key_body_err_rmg = torch.zeros((1, upper_dim), dtype=torch.float32, device=self.device)
        else:
            upper_key_body_err_rmg = upper_joint_err.repeat(1, rep)[:, :upper_dim]
        tracking_err_rmg = torch.cat((root_ang_vel_err_yaw, upper_key_body_err_rmg), dim=-1)
        arm_idx = self._upper_body_dof_idx[self._upper_body_dof_idx != self.waist_yaw_policy_idx]
        if arm_idx.size == 0:
            hand_proxy = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        else:
            arm_pos = (dof_pos_t[:, arm_idx] - self.default_dof_pos_t[:, arm_idx]) * self.cfg.obs_dof_pos_scale
            rep_h = (6 + arm_pos.shape[1] - 1) // arm_pos.shape[1]
            hand_proxy = arm_pos.repeat(1, rep_h)[:, :6]
        upper_body_rmg = torch.cat((waist_pos, waist_vel, base_ang_vel[:, :2], hand_proxy), dim=-1)
        proprio_current = torch.cat(
            (
                base_ang_vel * self.cfg.obs_ang_vel_scale,
                projected_gravity,
                (dof_pos_t - self.default_dof_pos_t) * self.cfg.obs_dof_pos_scale,
                dof_vel_t * self.cfg.obs_dof_vel_scale,
                self.action_tensor,
                tracking_err_rmg,
                upper_body_rmg,
            ),
            dim=-1,
        )
        for idx in self.cfg.ankle_idx:
            proprio_current[:, 6 + self.cfg.num_actions + idx] = 0.0

        mimic_obs = self._build_mimic_obs(
            torch.tensor([motion_time], device=self.device, dtype=torch.float32)
        )
        student_obs = torch.cat((mimic_obs, proprio_current), dim=-1)
        if student_obs.shape[-1] != N_OBS_SINGLE:
            raise ValueError(f"Expected student_obs dim {N_OBS_SINGLE}, got {student_obs.shape[-1]}")

        diag = {
            "base_quat_xyzw": q_raw.astype(np.float32),
            "base_quat_policy_xyzw": q_policy.astype(np.float32),
            "base_ang_vel_local": base_ang_vel[0].detach().cpu().numpy().astype(np.float32),
            "projected_gravity": projected_gravity[0].detach().cpu().numpy().astype(np.float32),
            "dof_pos_policy": dof_pos_t[0].detach().cpu().numpy().astype(np.float32),
            "dof_vel_policy": dof_vel_t[0].detach().cpu().numpy().astype(np.float32),
            "ref_dof_pos": ref_dof_pos_cur[0].detach().cpu().numpy().astype(np.float32),
            "mimic_obs": mimic_obs[0].detach().cpu().numpy().astype(np.float32),
        }
        return student_obs, diag

    @staticmethod
    def _assemble_policy_obs(student_obs: torch.Tensor, history_fill_mode: str) -> torch.Tensor:
        if history_fill_mode == "repeat_current":
            hist = student_obs.unsqueeze(1).expand(-1, HISTORY_LEN, -1).reshape(1, -1)
        elif history_fill_mode == "zeros":
            hist = torch.zeros((1, HISTORY_LEN * N_OBS_SINGLE), dtype=student_obs.dtype, device=student_obs.device)
        else:
            raise ValueError(f"unknown history_fill_mode: {history_fill_mode!r}")
        policy_obs = torch.cat((student_obs, hist), dim=-1)
        if policy_obs.shape[-1] != N_POLICY_OBS:
            raise ValueError(f"Expected policy obs dim {N_POLICY_OBS}, got {policy_obs.shape[-1]}")
        return policy_obs

    def _process_obs_sanity_actions(self, raw_action: np.ndarray, *, no_apply: bool) -> dict[str, np.ndarray]:
        clip_lim = self.cfg.clip_actions / self.action_scale
        clipped_action = np.clip(raw_action, -clip_lim, clip_lim).astype(np.float32)
        if self.cfg.action_lowpass_filter:
            alpha = float(self.cfg.action_filter_alpha)
            filtered_action = (alpha * clipped_action).astype(np.float32)
        else:
            filtered_action = clipped_action.copy()
        applied_action = self._apply_policy_action_ablations(self.cfg, filtered_action)
        actually_sent_action = np.zeros_like(applied_action) if no_apply else applied_action.copy()
        target_offset = actually_sent_action * self.action_scale
        dof_pos_target = (
            self._compute_balance_safe_pd_targets(actually_sent_action)
            if self.cfg.g1_balance_safe_enable
            else self.ref_dof_pos_res + target_offset
        )
        return {
            "raw_action": raw_action.astype(np.float32),
            "clipped_action": clipped_action,
            "filtered_action": filtered_action,
            "applied_action": applied_action,
            "actually_sent_action": actually_sent_action,
            "target_offset": target_offset.astype(np.float32),
            "dof_pos_target": dof_pos_target.astype(np.float32),
        }

    def run_obs_sanity_case(
        self,
        case: dict,
        case_index: int,
        *,
        ref_row: int,
        csv_fps: float,
        zero_vel: bool,
        use_real_dof: bool,
        use_real_last_action: bool,
        history_fill_mode: str,
        run_policy: bool,
        no_apply: bool,
        log_policy_input: bool,
    ) -> dict:
        ref_motion_time = self._motion_time_for_ref_row(
            int(case.get("ref_row", ref_row)),
            csv_fps,
        )
        self._set_obs_sanity_physics_state(
            case,
            ref_motion_time,
            zero_vel=zero_vel,
            use_real_dof=use_real_dof,
            use_real_last_action=use_real_last_action,
        )
        self._reset_yaw_align_state()
        student_obs, diag = self._build_student_obs_sanity(ref_motion_time)
        policy_obs = self._assemble_policy_obs(student_obs, history_fill_mode)
        obs_for_onnx = self._clip_raw_obs_for_onnx(policy_obs, self.cfg.clip_observations)

        action_pack = {
            "raw_action": np.zeros((self.cfg.num_actions,), dtype=np.float32),
            "clipped_action": np.zeros((self.cfg.num_actions,), dtype=np.float32),
            "filtered_action": np.zeros((self.cfg.num_actions,), dtype=np.float32),
            "applied_action": np.zeros((self.cfg.num_actions,), dtype=np.float32),
            "actually_sent_action": np.zeros((self.cfg.num_actions,), dtype=np.float32),
            "target_offset": np.zeros((self.cfg.num_actions,), dtype=np.float32),
            "dof_pos_target": diag["ref_dof_pos"].copy(),
        }
        if run_policy:
            raw_action = self._infer_onnx_actions(obs_for_onnx)
            action_pack = self._process_obs_sanity_actions(raw_action, no_apply=no_apply)

        motion_time_t = torch.tensor([ref_motion_time], device=self.device, dtype=torch.float32)
        root_pos, root_rot, root_vel, root_ang_vel, ref_dof_pos_t, ref_dof_vel_t, _ = self._calc_motion_frame(
            self.motion_ids, motion_time_t
        )
        query_t = ref_motion_time + float(self.cfg.mimic_tar_first_step) * self.dt
        max_t = float(self.motion_len - 0.01)
        query_t = min(query_t, max_t)
        t_future = min(query_t + float(self.cfg.ref_turn_intent_horizon_s), max_t)
        _, root_rot_future, _, _, _, _, _ = self.motion_lib.calc_motion_frame(
            self.motion_ids,
            torch.tensor([t_future], device=self.device, dtype=torch.float32),
        )
        heading_sin, heading_cos = ref_turn_intent_heading_sin_cos(root_rot, root_rot_future)
        ref_roll, ref_pitch, ref_yaw = euler_from_quaternion(root_rot)
        ref_vel_local = quat_rotate_inverse(root_rot, root_vel)
        ref_ang_local = quat_rotate_inverse(root_rot, root_ang_vel)
        _, _, ref_yaw_future = euler_from_quaternion(root_rot_future)

        bq = diag["base_quat_xyzw"]
        br, bp, by = _euler_xyz_from_quat_xyzw(bq)
        row = {
            "source": OBS_SANITY_SOURCE,
            "sanity_tag": str(case.get("sanity_tag", f"case_{case_index}")),
            "case_index": int(case_index),
            "ref_index": int(case.get("ref_row", ref_row)),
            "history_fill_mode": history_fill_mode,
            "imu_quat_w": float(bq[3]),
            "imu_quat_x": float(bq[0]),
            "imu_quat_y": float(bq[1]),
            "imu_quat_z": float(bq[2]),
            "imu_roll_rad": float(br),
            "imu_pitch_rad": float(bp),
            "imu_yaw_rad": float(by),
            "imu_roll_deg": float(np.rad2deg(br)),
            "imu_pitch_deg": float(np.rad2deg(bp)),
            "imu_yaw_deg": float(np.rad2deg(by)),
            "projected_gravity_x": float(diag["projected_gravity"][0]),
            "projected_gravity_y": float(diag["projected_gravity"][1]),
            "projected_gravity_z": float(diag["projected_gravity"][2]),
            "base_ang_vel_x": float(diag["base_ang_vel_local"][0]),
            "base_ang_vel_y": float(diag["base_ang_vel_local"][1]),
            "base_ang_vel_z": float(diag["base_ang_vel_local"][2]),
            "ref_root_pos_x": float(root_pos[0, 0].item()),
            "ref_root_pos_y": float(root_pos[0, 1].item()),
            "ref_root_pos_z": float(root_pos[0, 2].item()),
            "ref_roll": float(ref_roll[0].item()),
            "ref_pitch": float(ref_pitch[0].item()),
            "ref_yaw": float(ref_yaw[0].item()),
            "ref_vel_x": float(ref_vel_local[0, 0].item()),
            "ref_vel_y": float(ref_vel_local[0, 1].item()),
            "ref_vel_z": float(ref_vel_local[0, 2].item()),
            "ref_ang_vel_x": float(ref_ang_local[0, 0].item()),
            "ref_ang_vel_y": float(ref_ang_local[0, 1].item()),
            "ref_ang_vel_z": float(ref_ang_local[0, 2].item()),
            "ref_yaw_future": float(ref_yaw_future[0].item()),
            "ref_turn_intent_sin": float(heading_sin[0, 0].item()),
            "ref_turn_intent_cos": float(heading_cos[0, 0].item()),
        }
        ref_dof = diag["ref_dof_pos"]
        dof_pos = diag["dof_pos_policy"]
        dof_vel = diag["dof_vel_policy"]
        ref_dof_vel = ref_dof_vel_t[0].detach().cpu().numpy().astype(np.float32)
        for i in range(self.cfg.num_actions):
            row[f"ref_dof_pos_{i}"] = float(ref_dof[i])
            row[f"ref_dof_vel_{i}"] = float(ref_dof_vel[i])
            row[f"dof_pos_{i}"] = float(dof_pos[i])
            row[f"dof_vel_{i}"] = float(dof_vel[i])
            row[f"dof_pos_minus_ref_{i}"] = float(dof_pos[i] - ref_dof[i])
        for key in (
            "raw_action",
            "clipped_action",
            "filtered_action",
            "applied_action",
            "actually_sent_action",
            "target_offset",
            "dof_pos_target",
        ):
            vals = action_pack[key]
            for i in range(self.cfg.num_actions):
                row[f"{key}_{i}"] = float(vals[i])
        single_np = student_obs[0].detach().cpu().numpy().astype(np.float32)
        policy_np = obs_for_onnx[0].detach().cpu().numpy().astype(np.float32)
        for i in range(N_OBS_SINGLE):
            row[f"hth_single_obs_{i}"] = float(single_np[i])
        if log_policy_input:
            for i in range(N_POLICY_OBS):
                row[f"hth_policy_input_{i}"] = float(policy_np[i])
        return row

    def _reset_episode_buffers(self, *, zero_action: bool = True) -> None:
        self._reset_yaw_align_state()
        if zero_action:
            self.action_tensor.zero_()
        self.filtered_actions[:] = 0.0
        self.action_history_buf[:] = 0.0
        self._reset_action_delay_steps()
        self._prev_foot_contact_bool = [False, False]
        self._fall_detected_motion_time = None
        self._first_failure_time_s = None
        self._first_failure_reason = ""
        self.obs_history_buf.zero_()
        self.episode_length_buf.zero_()
        mujoco.mj_forward(self.model, self.data)
        _, _, by0 = _euler_xyz_from_quat_xyzw(np.array(self._get_base_quat_xyzw(), dtype=np.float64))
        self._init_yaw_diag = float(by0)

    def _reset_from_hth_rl_log(self) -> None:
        """no-reset / inject real pose：用真机 hth_rl_log 关节+IMU 初始化 MuJoCo，不走 ref[0] 姿态。"""
        row = load_hth_rl_log_row(self._inject_real_pose_csv, self._inject_real_pose_step)
        case = hth_rl_log_row_to_inject_case(row)
        if case.get("dof_pos") is None:
            raise ValueError(
                f"{self._inject_real_pose_csv!r} policy_step={self._inject_real_pose_step} "
                "缺少 dof_pos_* 列"
            )

        ref_time = float(self.motion_time_offset)
        if case.get("ref_time_s") is not None:
            ref_time = float(case["ref_time_s"])
        elif case.get("ref_row") is not None:
            ref_time = self._motion_time_for_ref_row(int(case["ref_row"]), self.csv_fps)

        motion_time_t = torch.tensor([ref_time], device=self.device, dtype=torch.float32)
        root_pos, root_rot, _, _, _, _, _ = self._calc_motion_frame(self.motion_ids, motion_time_t)

        dof_pos_policy = np.asarray(case["dof_pos"], dtype=np.float32).reshape(-1)
        if dof_pos_policy.size != self.cfg.num_actions:
            raise ValueError(
                f"inject dof_pos size {dof_pos_policy.size} != num_actions {self.cfg.num_actions}"
            )
        dof_pos_policy[self.left_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_left)
        dof_pos_policy[self.right_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_right)

        dof_vel_policy = np.zeros((self.cfg.num_actions,), dtype=np.float32)
        if self._inject_real_pose_use_dof_vel and case.get("dof_vel") is not None:
            dof_vel_policy = np.asarray(case["dof_vel"], dtype=np.float32).reshape(-1)

        self.data.qpos[self._act_qposadr] = self._policy_to_act_vec(dof_pos_policy)
        self.data.qvel[self._act_qveladr] = self._policy_to_act_vec(dof_vel_policy)

        rp = root_pos[0].detach().cpu().numpy().astype(np.float32)
        self.data.qpos[:3] = rp
        self.data.qpos[2] += float(self.cfg.root_height_offset + self.cfg.root_reset_z_offset)

        if (
            self._inject_real_pose_use_imu
            and all(k in case for k in ("imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z"))
        ):
            w = float(case["imu_quat_w"])
            x = float(case["imu_quat_x"])
            y = float(case["imu_quat_y"])
            z = float(case["imu_quat_z"])
            base_quat_xyzw = np.array([x, y, z, w], dtype=np.float32)
            nq = float(np.linalg.norm(base_quat_xyzw))
            if nq > 1e-8:
                base_quat_xyzw /= nq
            self.data.qpos[3] = float(base_quat_xyzw[3])
            self.data.qpos[4:7] = base_quat_xyzw[:3]
        else:
            self.data.qpos[3] = float(root_rot[0, 3].item())
            self.data.qpos[4:7] = root_rot[0, :3].detach().cpu().numpy()

        if self._inject_real_pose_zero_vel:
            self.data.qvel[:] = 0.0
        elif not self._inject_real_pose_use_dof_vel:
            self.data.qvel[self._act_qveladr] = 0.0

        last_action = np.zeros((self.cfg.num_actions,), dtype=np.float32)
        if self._inject_real_pose_last_action:
            for key in ("actually_sent_action", "applied_action", "filtered_action", "raw_action"):
                if case.get(key) is not None:
                    last_action = np.asarray(case[key], dtype=np.float32).reshape(-1)
                    break
        self.action_tensor[:] = torch.tensor(last_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        bq = self._get_base_quat_xyzw()
        br, bp, by = _euler_xyz_from_quat_xyzw(bq.astype(np.float64))
        print(
            f"[inject-real-pose] csv={os.path.abspath(self._inject_real_pose_csv)} "
            f"policy_step={self._inject_real_pose_step} ref_time={ref_time:.3f}s "
            f"imu_rpy=({np.degrees(br):.1f},{np.degrees(bp):.1f},{np.degrees(by):.1f})° "
            f"dof_pos[0:3]={dof_pos_policy[:3]} zero_vel={self._inject_real_pose_zero_vel} "
            f"last_action={'yes' if self._inject_real_pose_last_action else 'no'}",
            flush=True,
        )

    def reset(self):
        if self._inject_real_pose_csv:
            self._reset_from_hth_rl_log()
            self._reset_episode_buffers(zero_action=not self._inject_real_pose_last_action)
            return

        motion_time = torch.tensor([self.motion_time_offset], device=self.device, dtype=torch.float32)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, _ = self._calc_motion_frame(
            self.motion_ids, motion_time
        )
        dof_pos_policy = dof_pos[0].detach().cpu().numpy().astype(np.float32)
        dof_pos_policy[self.left_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_left)
        dof_pos_policy[self.right_ankle_pitch_policy_idx] += float(self.cfg.ankle_pitch_bias_right)
        dof_vel_policy = (dof_vel[0] * self.cfg.reset_vel_factor).detach().cpu().numpy().astype(np.float32)
        if self.cfg.yaw_align_to_ref:
            # 真机 STAND→WALK：进入策略前近似静止，避免 ref 帧速度与世界系朝向不一致
            dof_vel_policy[:] = 0.0
        self.data.qpos[self._act_qposadr] = self._policy_to_act_vec(dof_pos_policy)
        self.data.qvel[self._act_qveladr] = self._policy_to_act_vec(dof_vel_policy)
        self.data.qpos[:3] = root_pos[0].detach().cpu().numpy()
        self.data.qpos[2] += self.cfg.root_height_offset + self.cfg.root_reset_z_offset
        if self.cfg.yaw_align_to_ref and self.cfg.yaw_align_random_yaw:
            ref_q = root_rot[0].detach().cpu().numpy().astype(np.float64)
            ref_yaw = _yaw_from_quat_xyzw(ref_q)
            max_rad = float(np.deg2rad(max(0.0, self.cfg.yaw_align_max_deg)))
            delta_yaw = float(np.random.uniform(-max_rad, max_rad))
            random_yaw = _wrap_to_pi(ref_yaw + delta_yaw)
            base_quat_xyzw = _apply_yaw_align_xyzw(ref_q, random_yaw - ref_yaw)
            self.data.qpos[3] = float(base_quat_xyzw[3])
            self.data.qpos[4:7] = base_quat_xyzw[:3]
            print(
                f"[yaw_align] reset random yaw={np.degrees(random_yaw):.1f}° "
                f"(ref={np.degrees(ref_yaw):.1f}°, Δ={np.degrees(delta_yaw):.1f}°, "
                f"max=±{self.cfg.yaw_align_max_deg:.0f}°)",
                flush=True,
            )
        else:
            self.data.qpos[3] = float(root_rot[0, 3].item())
            self.data.qpos[4:7] = root_rot[0, :3].detach().cpu().numpy()
        if self.cfg.yaw_align_to_ref:
            self.data.qvel[:] = 0.0
        else:
            self.data.qvel[:3] = (root_vel[0] * self.cfg.reset_vel_factor).detach().cpu().numpy().astype(np.float32)
            root_quat_xyzw = root_rot[0].detach().cpu().numpy().astype(np.float64)
            root_ang_vel_world = root_ang_vel[0].detach().cpu().numpy().astype(np.float64)
            root_ang_vel_local = world_ang_vel_to_local_xyzw(
                root_quat_xyzw,
                root_ang_vel_world,
            )
            self.data.qvel[3:6] = (
                np.asarray(root_ang_vel_local, dtype=np.float32)
                * float(self.cfg.reset_vel_factor)
            )
        self._reset_episode_buffers()

    def _reset_action_delay_steps(self) -> None:
        if self.cfg.action_delay_steps is not None:
            delay = int(self.cfg.action_delay_steps)
        elif self.cfg.action_delay:
            delay = int(np.random.randint(0, self.cfg.ctrl_delay_step_range + 1))
        else:
            delay = 0
        max_delay = max(0, min(delay, self.cfg.action_buf_len - 1))
        self._action_delay_steps = max_delay

    def _postprocess_policy_action(self, action: np.ndarray) -> np.ndarray:
        """与 cp_fdd_V1.step 相同顺序：clip 后低通 → 写入 history → 按 delay 取指令。"""
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.cfg.action_lowpass_filter:
            alpha = float(self.cfg.action_filter_alpha)
            a = (1.0 - alpha) * self.filtered_actions + alpha * a
            self.filtered_actions[:] = a
        if self.action_history_buf.shape[0] > 1:
            self.action_history_buf[:-1] = self.action_history_buf[1:]
        self.action_history_buf[-1] = a
        if self.cfg.action_delay:
            idx = -(self._action_delay_steps + 1)
            a = self.action_history_buf[idx].copy()
        return self._apply_policy_action_ablations(self.cfg, a)

    def _step_policy(self, motion_time_clamped_tensor, unclamped_motion_time: float):
        log_pre = None
        if self._inference_log_writer is not None:
            base_quat_xyzw = self._get_base_quat_xyzw()
            base_quat_t = torch.tensor(base_quat_xyzw, dtype=torch.float32, device=self.device).unsqueeze(0)
            base_ang_vel_local = self._get_base_ang_vel_local(base_quat_t)[0].detach().cpu().numpy().astype(np.float32)
            dof_pos_act0, dof_vel_act0 = self._get_dof_state_act()
            log_pre = {
                "timestamp": float(unclamped_motion_time),
                "base_quat_xyzw": base_quat_xyzw,
                "base_ang_vel_local": base_ang_vel_local,
                "joint_pos_policy": self._act_to_policy_vec(dof_pos_act0),
                "joint_vel_policy": self._act_to_policy_vec(dof_vel_act0),
            }
        raw_obs = self._build_actor_obs(motion_time_clamped_tensor)
        obs_for_onnx = self._clip_raw_obs_for_onnx(raw_obs, self.cfg.clip_observations)
        raw_action = self._infer_onnx_actions(obs_for_onnx)
        clip_lim = self.cfg.clip_actions / self.action_scale
        clipped_action = np.clip(raw_action, -clip_lim, clip_lim)
        action_cmd = self._postprocess_policy_action(clipped_action)
        action_eff = action_cmd
        self.action_tensor[:] = torch.tensor(action_cmd, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Isaac: actions_scaled = actions * action_scale + ref_dof_pos（v4 另加 offset clip / limit clamp）
        target_dof_pos_policy = self._policy_action_to_dof_pos_target(action_cmd)
        target_dof_pos_act = self._policy_to_act_vec(target_dof_pos_policy)
        for _ in range(self.cfg.decimation):
            dof_pos_act, dof_vel_act = self._get_dof_state_act()
            tau_raw = (target_dof_pos_act - dof_pos_act) * self.kps_act - dof_vel_act * self.kds_act
            tau = self._limit_torques(tau_raw, dof_vel_act)
            self.data.ctrl[:] = tau
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
                self._jitter_diag.record_after_step(self.data, action_cmd, target_dof_pos_policy, track_ctx=ctx)

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
                raw_policy_obs=raw_obs[0].detach().cpu().numpy(),
            )

    def _limit_torques(self, tau, dof_vel_act):
        if self.cfg.enable_tn_limit:
            omega_nl = np.maximum(self.motor_omega_nl, 1e-6)
            dyn_cap = self.motor_tau_stall * np.maximum(1.0 - np.abs(dof_vel_act) / omega_nl, 0.0)
            if self.cfg.power_limit is not None:
                power_limit = np.asarray(self.cfg.power_limit, dtype=np.float32)
                if power_limit.size == 1:
                    power_limit = np.full((self.cfg.num_actions,), float(power_limit.item()), dtype=np.float32)
                if power_limit.size == self.cfg.num_actions:
                    dyn_cap = np.minimum(dyn_cap, self._policy_to_act_vec(power_limit) / (np.abs(dof_vel_act) + 1e-6))
            cap = np.minimum(dyn_cap, self.torque_limits)
        else:
            cap = self.torque_limits
        return np.clip(tau, -cap, cap)

    def _run_sim_loop(self, headless: bool, viewer=None) -> None:
        while True:
            viewer_running = True if headless else bool(viewer.is_running())
            step_start = time.time()
            u_t = self.motion_time_offset + float(self.episode_length_buf.item()) * self.dt
            self._last_motion_time = float(u_t)
            if not self._should_continue_run(u_t, viewer_running):
                break
            motion_time = min(u_t, self.motion_len - 0.01)
            self._step_policy(
                torch.tensor([motion_time], device=self.device, dtype=torch.float32),
                u_t,
            )
            self.episode_length_buf += 1
            if self._should_stop_run_after_fall(u_t):
                fallen, reason = self._fall_termination_reason()
                self._run_stop_reason = reason or "fall"
                print(
                    f"[sim2sim] fall termination after {FALL_TERMINATION_DELAY_S}s "
                    f"(motion_time={u_t:.2f}, reason={self._run_stop_reason})",
                    flush=True,
                )
                break
            u_t_after = self.motion_time_offset + float(self.episode_length_buf.item()) * self.dt
            self._last_motion_time = float(u_t_after)
            if not self._should_continue_run(u_t_after, viewer_running):
                self._print_run_stop_reason(u_t_after)
                break
            if not headless:
                viewer.sync()
            sleep_t = self.dt - (time.time() - step_start)
            if sleep_t > 0.0:
                time.sleep(sleep_t)

    def run(self, headless=False):
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
        try:
            if headless:
                self._run_sim_loop(headless=True)
            else:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    self._run_sim_loop(headless=False, viewer=viewer)
        finally:
            self._close_inference_log()
            self._write_run_metadata()
            if self._jitter_diag is not None:
                self._jitter_diag.finalize()


def run_obs_sanity_mode(args: argparse.Namespace) -> None:
    labels = build_cp_fdd_v1_single_obs_labels()
    assert len(labels) == N_OBS_SINGLE
    label_path = os.path.abspath(args.obs_sanity_label_output)
    label_dir = os.path.dirname(label_path)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for line in labels:
            f.write(f"{line}\n")
    print(f"[obs-sanity] wrote {len(labels)} single-obs labels -> {label_path}")

    cases = resolve_obs_sanity_cases(args)
    history_fill_mode = str(args.obs_sanity_history_fill_mode)
    log_policy_input = bool(args.obs_sanity_log_policy_input)

    cfg_overrides: dict = {
        "ref_turn_intent_horizon_s": float(args.ref_turn_intent_horizon_s),
        "ankle_pitch_bias_left": float(args.ankle_pitch_bias_left),
        "ankle_pitch_bias_right": float(args.ankle_pitch_bias_right),
        "inference_log_csv": None,
        "yaw_align_to_ref": bool(args.yaw_align_to_ref),
        "yaw_align_max_deg": float(args.yaw_align_max_deg),
        "yaw_align_full_proprio": not bool(args.yaw_align_cpp_compat),
        "yaw_align_random_yaw": bool(args.yaw_align_random_yaw),
    }
    if args.obs_sanity_no_action_lowpass or args.no_action_lowpass:
        cfg_overrides["action_lowpass_filter"] = False
        cfg_overrides["action_filter_alpha"] = 1.0
    elif args.action_filter_alpha is not None:
        cfg_overrides["action_filter_alpha"] = float(args.action_filter_alpha)
    if args.no_action_delay:
        cfg_overrides["action_delay"] = False

    cfg = make_rmg_config(
        device=args.device,
        sim_duration=1.0,
        v1_clean=bool(args.v1_clean),
        hmft=bool(args.hmft),
        v9ft=bool(args.v9ft),
        amass_startup=bool(args.amass_startup),
        **cfg_overrides,
    )
    sim = MujocoRMGSim(
        args.xml,
        args.onnx,
        args.motion_csv,
        args.csv_fps,
        cfg,
        skip_onnx_normalizer_check=bool(args.skip_onnx_normalizer_check),
    )
    sim.reset()

    out_path = os.path.abspath(args.obs_sanity_output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = obs_sanity_csv_fieldnames(sim.cfg.num_actions, log_policy_input)
    rows: list[dict] = []
    for case_index, case in enumerate(cases):
        row = sim.run_obs_sanity_case(
            case,
            case_index,
            ref_row=int(args.obs_sanity_ref_row),
            csv_fps=float(args.csv_fps),
            zero_vel=bool(args.obs_sanity_zero_vel),
            use_real_dof=bool(args.obs_sanity_use_real_dof),
            use_real_last_action=bool(args.obs_sanity_use_real_last_action),
            history_fill_mode=history_fill_mode,
            run_policy=bool(args.obs_sanity_run_policy),
            no_apply=bool(args.obs_sanity_no_apply),
            log_policy_input=log_policy_input,
        )
        rows.append(row)
        print(
            f"[obs-sanity] case {case_index} tag={row['sanity_tag']!r} "
            f"imu_pitch_deg={row['imu_pitch_deg']:.3f} "
            f"proj_g=({row['projected_gravity_x']:.4f},{row['projected_gravity_y']:.4f},{row['projected_gravity_z']:.4f})"
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[obs-sanity] wrote {len(rows)} rows -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo ONNX inference for cp_rmg (CSV direct input)")
    parser.add_argument("--xml", default=f"{PRJ_DIR}/../assets/casbot_02/xml/CASBOT02_ENCOS_5dof_skeleton_20250904.xml")
    parser.add_argument(
        "--onnx",
        default=f"{PRJ_DIR}/config/onnx/model_51500_actor_repair_05161332.onnx",
        help="须为 save_cp_stu --with_normalizer 导出的 ONNX（图内含 mean/std 归一化）",
    )
    parser.add_argument(
        "--skip-onnx-normalizer-check",
        action="store_true",
        help="跳过 ONNX 内嵌 normalizer 检测（纯 actor 且自行在外部 normalize 时使用）",
    )
    parser.add_argument("--motion-csv", default=f"{PRJ_DIR}/config/by_csv_finetune/stand.csv")
    parser.add_argument("--csv-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=None,
        help=(
            "沿参考动作时间轴的播放时长（秒），从 --start-time 起算；"
            "未指定则播放到动作结束。倒地仍按原逻辑提前终止。"
        ),
    )
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument(
        "--ref-turn-intent-horizon-s",
        type=float,
        default=REF_TURN_INTENT_HORIZON_S,
        help="mimic 航向槽参考未来转向意图时间窗（秒），与 cp_fdd_V1 ref_turn_intent_horizon_s 一致",
    )
    parser.add_argument(
        "--turn-yaw-scale",
        type=float,
        default=1.0,
        help=(
            "转向动作参考 yaw 角速度缩放（仅对 YuanDiZhuan_You/Zuo 等子串匹配 motion 生效，"
            "对齐 Isaac turn_curriculum end_scale；1.0=不缩放）"
        ),
    )
    parser.add_argument(
        "--turn-yaw-scale-motion-substrings",
        type=str,
        default=",".join(DEFAULT_TURN_YAW_SCALE_MOTION_SUBSTRINGS),
        help="逗号分隔；motion CSV 文件名含任一子串时应用 --turn-yaw-scale",
    )
    parser.add_argument(
        "--mimic-ref-wz-scaled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="metadata：mimic ref_wz 是否经 turn_yaw_scale 缩放（默认随 turn_yaw_scale!=1）",
    )
    parser.add_argument(
        "--ref-end-hold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ref_end 后固定末帧姿态并继续推理 hold 秒数（v6 hold5 评测）",
    )
    parser.add_argument(
        "--ref-end-hold-s",
        type=float,
        default=5.0,
        help="ref_end 后 hold 时长（秒），仅 --ref-end-hold 时生效",
    )
    parser.add_argument(
        "--max-episode-length-s",
        type=float,
        default=None,
        help="全局 episode 上界（秒）；长动作未到 ref_end 时记 horizon_truncated",
    )
    parser.add_argument(
        "--ankle-pitch-bias-left",
        type=float,
        default=0.0,
        help="左踝 pitch 持续角度偏置(弧度)，叠加到每步关节目标角",
    )
    parser.add_argument(
        "--ankle-pitch-bias-right",
        type=float,
        default=0.0,
        help="右踝 pitch 持续角度偏置(弧度)，叠加到每步关节目标角",
    )
    parser.add_argument(
        "--clip-actions",
        type=float,
        default=None,
        help="策略 action 对称裁剪限幅（与 cp normalization.clip_actions 一致）；未指定时用 CP_RMG_V1 默认",
    )
    parser.add_argument(
        "--v1-clean",
        action="store_true",
        help="使用 v1_clean 基线：clip_actions=5、关闭低通与 action delay",
    )
    parser.add_argument(
        "--amass-startup",
        action="store_true",
        help=(
            "C306RMGRefTurnIntentAMASSStartupCfg 部署预设：clip=0.6、无低通/无 delay、"
            "init_state 腿关节 default=0"
        ),
    )
    parser.add_argument(
        "--hmft",
        action="store_true",
        help="HMFT v4/v5/v6 部署预设：clip=0.45、无 delay、低通 alpha=0.4、action_buf_len=2",
    )
    parser.add_argument(
        "--v9ft",
        action="store_true",
        help="HMFT v9ft 部署预设：clip/lowpass=0.60、无 delay、下肢 PD kp×1.10 kd×1.102",
    )
    parser.add_argument(
        "--action-filter-alpha",
        "--action-lowpass-alpha",
        type=float,
        default=None,
        help="低通 alpha / action_filter_new_weight（默认 0.4）",
    )
    parser.add_argument(
        "--full-ref",
        action="store_true",
        help="沿参考轨迹 full-ref 评测：eval 上限=ref duration（不施加外部短 cap）",
    )
    parser.add_argument(
        "--ref-duration-s",
        type=float,
        default=None,
        help="参考轨迹时长（秒）；默认由 motion CSV 行数与 csv_fps 推断",
    )
    parser.add_argument(
        "--eval-duration-s",
        type=float,
        default=None,
        help="本次评测终止上限（秒）；full-ref 时默认等于 ref duration",
    )
    parser.add_argument(
        "--run-metadata-json",
        type=str,
        default=None,
        help="写入 rollout 实际 action/eval 配置的 JSON 路径",
    )
    parser.add_argument(
        "--no-action-lowpass",
        action="store_true",
        help="关闭 action 低通（覆盖 RealRepair 默认）",
    )
    parser.add_argument(
        "--no-action-delay",
        action="store_true",
        help="关闭 action delay（覆盖 RealRepair 默认）",
    )
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=None,
        help="固定 delay 步数 0~ctrl_delay_step_range；默认每回合随机 0~2",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--zero-policy-action",
        action="store_true",
        help="策略照常 forward，在构造 PD target 之前将 residuals 全部置零（消融：资产/参考/PD 默认值）",
    )
    parser.add_argument(
        "--zero-ankle-roll-action",
        action="store_true",
        help="仅屏蔽踝 roll 的 residual（见脚本顶部 ANKLE_ROLL_ACTION_INDICES）",
    )
    parser.add_argument(
        "--zero-hip-roll-and-ankle-roll-action",
        action="store_true",
        help="同时屏蔽髋 roll 与踝 roll residuals",
    )
    parser.add_argument(
        "--diagnose-jitter-slip",
        action="store_true",
        help="启用每步物理仿真的抖动/滑移诊断（略增开销，运行结束写入 diagnostics 目录）",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default=None,
        help="诊断输出父目录；默认 tw_cp_mujoco_119/diagnostics/<时间戳>/",
    )
    parser.add_argument(
        "--jitter-cutoff-hz",
        type=float,
        default=25.0,
        help="高通参考截止频率 Hz（butter/EMA 低频估计）；默认 25",
    )
    parser.add_argument(
        "--diagnose-sim2sim",
        action="store_true",
        help="与 --diagnose-jitter-slip 相同，启用完整 sim2sim 日志（速度/踝/足姿态等）",
    )
    parser.add_argument(
        "--diagnostics-note",
        type=str,
        default="",
        help="写入 gptpro_report.md / run_metadata 的备注",
    )
    parser.add_argument(
        "--inference-log-csv",
        type=str,
        default=None,
        help="若指定路径，则每个控制步将 timestamp/IMU/关节/动作/参考状态写入 CSV（与策略推理同频）",
    )
    parser.add_argument(
        "--inference-log-policy-obs",
        action="store_true",
        help="sim2sim inference log 额外输出 hth_single_obs_0..124",
    )
    parser.add_argument(
        "--inference-log-full-policy-input",
        action="store_true",
        help="与 --inference-log-policy-obs 联用，额外输出 hth_policy_input_0..1374",
    )
    parser.add_argument("--obs-sanity", action="store_true", help="启用 HTH obs/action 单帧对照模式（不推进 rollout）")
    parser.add_argument("--obs-sanity-output", default="sim2sim_obs_sanity_log.csv", type=str)
    parser.add_argument("--obs-sanity-label-output", default="sim2sim_single_obs_labels.txt", type=str)
    parser.add_argument("--obs-sanity-real-log", default=None, type=str, help="真机 hth_obs_sanity_log.csv，用于抽取姿态样本")
    parser.add_argument(
        "--obs-sanity-tags",
        default=None,
        type=str,
        help="逗号分隔 sanity_tag，例如 neutral_hold,manual_backward_tilt,...",
    )
    parser.add_argument(
        "--obs-sanity-manual-poses",
        default=None,
        type=str,
        help='手动姿态：\"neutral:0,0,0;backward:0,-8,0\"（deg，roll,pitch,yaw）',
    )
    parser.add_argument("--obs-sanity-ref-row", default=0, type=int, help="stand.csv 参考帧行号")
    parser.add_argument("--obs-sanity-zero-vel", default=True, type=_str2bool, help="base 线/角速度置零")
    parser.add_argument("--obs-sanity-run-policy", default=True, type=_str2bool, help="对构造 obs 运行 ONNX")
    parser.add_argument(
        "--obs-sanity-no-apply",
        default=True,
        type=_str2bool,
        help="只算 policy action，actually_sent_action 置零（对齐真机 no_apply）",
    )
    parser.add_argument("--obs-sanity-use-real-dof", default=False, type=_str2bool, help="使用真机 log 中的 dof_pos/dof_vel")
    parser.add_argument(
        "--obs-sanity-use-real-last-action",
        default=False,
        type=_str2bool,
        help="使用真机 log 中的 applied/actually_sent/raw action 作为 last_action",
    )
    parser.add_argument(
        "--obs-sanity-history-fill-mode",
        default="repeat_current",
        choices=("repeat_current", "zeros"),
        help="policy 1375 维 history 填充方式",
    )
    parser.add_argument(
        "--obs-sanity-log-policy-input",
        default=True,
        type=_str2bool,
        help="CSV 是否写入 hth_policy_input_0..1374",
    )
    parser.add_argument(
        "--obs-sanity-no-action-lowpass",
        action="store_true",
        help="obs sanity 单帧关闭 action 低通（prev=0 等价于 filtered=alpha*clipped）",
    )
    parser.add_argument(
        "--yaw-align-to-ref",
        action="store_true",
        help="obs 层 yaw 欺骗：projected_gravity（+ 默认 full_proprio）对齐 ref[0] 航向，"
        "不改 MuJoCo 物理朝向；真机 STAND→WALK 复现",
    )
    parser.add_argument(
        "--yaw-align-random-yaw",
        action="store_true",
        help="sim 压力测试：reset 时额外随机物理 yaw（需 --yaw-align-to-ref）；"
        "默认只欺骗 obs、不改物理",
    )
    parser.add_argument(
        "--yaw-align-max-deg",
        type=float,
        default=180.0,
        help="--yaw-align-random-yaw 时 reset 相对 ref[0] 的随机 yaw 半宽（度），默认 ±180°",
    )
    parser.add_argument(
        "--yaw-align-cpp-compat",
        action="store_true",
        help="与 C++ 真机一致：仅 projected_gravity 用 q_policy，base_ang_vel 仍用 raw IMU",
    )
    parser.add_argument(
        "--inject-real-pose-csv",
        type=str,
        default=None,
        help="no-reset 实验：用真机 hth_rl_log.csv 指定 policy_step 的 dof_pos/IMU 初始化 MuJoCo，"
        "不走 ref[0] reset",
    )
    parser.add_argument(
        "--inject-real-pose-step",
        type=int,
        default=0,
        help="--inject-real-pose-csv 使用的 policy_step（默认 0；无 policy_step 列则当行号）",
    )
    parser.add_argument(
        "--inject-real-pose-no-imu",
        action="store_true",
        help="inject 时不写 IMU 四元数，根姿态仍用 ref motion",
    )
    parser.add_argument(
        "--inject-real-pose-use-dof-vel",
        action="store_true",
        help="inject 时写入真机 dof_vel_*（默认关节速度置零）",
    )
    parser.add_argument(
        "--inject-real-pose-last-action",
        action="store_true",
        help="inject 时将 action_tensor 设为 log 中 actually_sent/applied action",
    )
    parser.add_argument(
        "--inject-real-pose-keep-vel",
        action="store_true",
        help="inject 时保留已写入的 qvel（默认清零全部速度）",
    )
    parser.add_argument(
        "--obs-inject-real-log",
        type=str,
        default=None,
        help="真机 hth_rl_log CSV 路径，用于 policy obs 上半身误差/速度注入",
    )
    parser.add_argument(
        "--obs-inject-upper-error",
        action="store_true",
        help="开启上半身 dof_pos 误差注入（仅 policy obs，不改物理/PD）",
    )
    parser.add_argument(
        "--obs-inject-upper-vel",
        action="store_true",
        help="开启上半身 dof_vel 注入（仅 policy obs，不改物理/PD）",
    )
    parser.add_argument(
        "--obs-inject-upper-scale",
        type=float,
        default=1.0,
        help="对注入的真机上半身误差/速度做缩放（敏感性测试，如 0.5/1.0/1.5）",
    )
    parser.add_argument(
        "--obs-inject-time-col",
        type=str,
        default="auto",
        choices=("auto", "ref_time_s", "global_time_s", "timestamp"),
        help="真机 log 插值时间列；auto 优先 ref_time_s > global_time_s > timestamp",
    )
    parser.add_argument(
        "--obs-inject-time-shift-s",
        type=float,
        default=0.0,
        help="插值时使用 log_time = motion_time + shift（秒）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.obs_sanity:
        run_obs_sanity_mode(args)
        return
    cfg_overrides: dict = {
        "ref_turn_intent_horizon_s": float(args.ref_turn_intent_horizon_s),
        "ankle_pitch_bias_left": float(args.ankle_pitch_bias_left),
        "ankle_pitch_bias_right": float(args.ankle_pitch_bias_right),
        "zero_policy_action": bool(args.zero_policy_action),
        "zero_ankle_roll_action": bool(args.zero_ankle_roll_action),
        "zero_hip_roll_and_ankle_roll_action": bool(args.zero_hip_roll_and_ankle_roll_action),
        "inference_log_csv": args.inference_log_csv,
        "yaw_align_to_ref": bool(args.yaw_align_to_ref),
        "yaw_align_max_deg": float(args.yaw_align_max_deg),
        "yaw_align_full_proprio": not bool(args.yaw_align_cpp_compat),
        "yaw_align_random_yaw": bool(args.yaw_align_random_yaw),
        "obs_inject_real_log": args.obs_inject_real_log,
        "obs_inject_upper_error": bool(args.obs_inject_upper_error),
        "obs_inject_upper_vel": bool(args.obs_inject_upper_vel),
        "obs_inject_upper_scale": float(args.obs_inject_upper_scale),
        "obs_inject_time_col": str(args.obs_inject_time_col),
        "obs_inject_time_shift_s": float(args.obs_inject_time_shift_s),
        "inference_log_policy_obs": bool(getattr(args, "inference_log_policy_obs", False)),
        "inference_log_full_policy_input": bool(getattr(args, "inference_log_full_policy_input", False)),
    }
    if args.clip_actions is not None:
        cfg_overrides["clip_actions"] = float(args.clip_actions)
    if args.no_action_lowpass:
        cfg_overrides["action_lowpass_filter"] = False
        cfg_overrides["action_filter_alpha"] = 1.0
    elif args.action_filter_alpha is not None:
        cfg_overrides["action_filter_alpha"] = float(args.action_filter_alpha)
    if args.no_action_delay:
        cfg_overrides["action_delay"] = False
    elif args.action_delay_steps is not None:
        cfg_overrides["action_delay"] = True
        cfg_overrides["action_delay_steps"] = int(args.action_delay_steps)
    cfg = make_rmg_config(
        device=args.device,
        sim_duration=args.sim_duration,
        v1_clean=bool(args.v1_clean),
        hmft=bool(args.hmft),
        v9ft=bool(args.v9ft),
        amass_startup=bool(args.amass_startup),
        **cfg_overrides,
    )
    sim = MujocoRMGSim(
        args.xml,
        args.onnx,
        args.motion_csv,
        args.csv_fps,
        cfg,
        skip_onnx_normalizer_check=bool(args.skip_onnx_normalizer_check),
    )
    if args.inject_real_pose_csv:
        sim._inject_real_pose_csv = os.path.abspath(args.inject_real_pose_csv)
        sim._inject_real_pose_step = int(args.inject_real_pose_step)
        sim._inject_real_pose_use_imu = not bool(args.inject_real_pose_no_imu)
        sim._inject_real_pose_use_dof_vel = bool(args.inject_real_pose_use_dof_vel)
        sim._inject_real_pose_last_action = bool(args.inject_real_pose_last_action)
        sim._inject_real_pose_zero_vel = not bool(args.inject_real_pose_keep_vel)
        inject_row = load_hth_rl_log_row(sim._inject_real_pose_csv, sim._inject_real_pose_step)
        inject_case = hth_rl_log_row_to_inject_case(inject_row)
        if inject_case.get("ref_time_s") is not None:
            sim.motion_time_offset = float(inject_case["ref_time_s"])
        elif inject_case.get("ref_row") is not None:
            sim.motion_time_offset = sim._motion_time_for_ref_row(
                int(inject_case["ref_row"]), float(args.csv_fps)
            )
    sim.turn_yaw_scale = float(args.turn_yaw_scale)
    sim._turn_yaw_scale_motion_substrings = tuple(
        s.strip() for s in str(args.turn_yaw_scale_motion_substrings).split(",") if s.strip()
    ) or DEFAULT_TURN_YAW_SCALE_MOTION_SUBSTRINGS
    sim.mimic_ref_wz_scaled = (
        bool(args.mimic_ref_wz_scaled)
        if args.mimic_ref_wz_scaled is not None
        else sim._turn_yaw_scale_applies()
    )
    sim.ref_pose_scaled = False
    sim.ref_end_hold_enabled = bool(args.ref_end_hold)
    sim.ref_end_hold_s = float(args.ref_end_hold_s)
    sim.zero_ref_vel_after_end = True
    sim.reset_after_hold_if_alive = True
    sim.max_episode_length_s = (
        float(args.max_episode_length_s) if args.max_episode_length_s is not None else None
    )
    if sim.ref_end_hold_enabled and sim.max_episode_length_s is None:
        sim.max_episode_length_s = 50.0
    if sim._turn_yaw_scale_applies():
        print(
            f"[sim2sim] turn_yaw_scale={sim.turn_yaw_scale:.3f} "
            f"on {os.path.basename(sim._motion_csv_path)} "
            f"(ref yaw rate × scale; ref pose/quat unchanged)",
            flush=True,
        )
    if args.inject_real_pose_csv:
        sim.motion_time_offset = max(0.0, min(sim.motion_time_offset, sim.motion_len - 0.05))
        if args.start_time != 0.0:
            sim.motion_time_offset = max(0.0, min(float(args.start_time), sim.motion_len - 0.05))
    else:
        sim.motion_time_offset = max(0.0, min(args.start_time, sim.motion_len - 0.05))
    ref_duration_s = (
        float(args.ref_duration_s)
        if args.ref_duration_s is not None
        else compute_ref_motion_duration_s(args.motion_csv, float(args.csv_fps))
    )
    sim.ref_duration_s = min(float(ref_duration_s), float(sim.motion_len))
    if args.full_ref:
        sim.full_ref = True
        sim.eval_duration_s = (
            float(args.eval_duration_s)
            if args.eval_duration_s is not None
            else float(sim.ref_duration_s)
        )
        if args.sim_duration is not None:
            print(
                f"[sim2sim] WARN: --full-ref 忽略 --sim-duration={args.sim_duration}",
                flush=True,
            )
        sim.cfg.sim_duration = None
    elif args.eval_duration_s is not None:
        sim.eval_duration_s = float(args.eval_duration_s)
        sim.cfg.sim_duration = float(args.eval_duration_s)
    else:
        sim.eval_duration_s = (
            float(sim.cfg.sim_duration)
            if sim.cfg.sim_duration is not None
            else float(sim.ref_duration_s)
        )
    sim.run_metadata_json = (
        os.path.abspath(args.run_metadata_json) if args.run_metadata_json else None
    )
    if args.diagnose_jitter_slip or args.diagnose_sim2sim:
        parent = args.diagnostics_dir or os.path.join(PRJ_DIR, "diagnostics")
        out_dir = os.path.join(parent, time.strftime("%Y%m%d-%H%M%S"))
        import sys

        meta = {
            "argv": sys.argv,
            "xml": os.path.abspath(args.xml),
            "onnx": os.path.abspath(args.onnx),
            "motion_csv": os.path.abspath(args.motion_csv),
            "csv_fps": args.csv_fps,
            "note": args.diagnostics_note or None,
            "zero_policy_action": cfg.zero_policy_action,
            "zero_ankle_roll_action": cfg.zero_ankle_roll_action,
            "zero_hip_roll_and_ankle_roll_action": cfg.zero_hip_roll_and_ankle_roll_action,
            "ankle_roll_action_indices": ANKLE_ROLL_ACTION_INDICES,
            "hip_roll_action_indices": HIP_ROLL_ACTION_INDICES,
            "sim_duration": cfg.sim_duration,
        }
        jcfg = JitterSlipDiagnosticsConfig(
            physics_dt=cfg.simulation_dt,
            output_dir=out_dir,
            jitter_cutoff_hz=float(args.jitter_cutoff_hz),
            ankle_policy_indices=tuple(int(x) for x in cfg.ankle_idx),
            run_metadata=meta,
        )
        sim._jitter_diag = JitterSlipDiagnostics(sim.model, jcfg)
    sim.run(headless=args.headless)


if __name__ == "__main__":
    main()
