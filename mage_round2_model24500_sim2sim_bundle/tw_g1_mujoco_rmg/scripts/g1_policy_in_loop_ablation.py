#!/usr/bin/env python3
"""
G1 policy-in-loop sim2sim ablation utilities: joint groups, control trace, diagnostics summary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# G1 21-DoF policy order (must match g1_mujoco_sim_rmg_onnx_csv.G1_POLICY_DOF_NAMES)
G1_POLICY_DOF_NAMES: Tuple[str, ...] = (
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
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

N_OBS_SINGLE = 125
HISTORY_LEN = 10
N_POLICY_OBS = N_OBS_SINGLE * (HISTORY_LEN + 1)


@dataclass
class ObsFieldSpec:
    name: str
    start: int
    end: int
    description: str


def build_g1_obs_field_registry() -> Dict[str, ObsFieldSpec]:
    """single_obs=125 layout: mimic(30) + proprio(95)."""
    mimic = 30
    p0 = mimic
    fields = {
        "mimic": ObsFieldSpec("mimic", 0, mimic, "reference mimic obs"),
        "base_ang_vel": ObsFieldSpec("base_ang_vel", p0, p0 + 3, "scaled base angular velocity"),
        "projected_gravity": ObsFieldSpec("projected_gravity", p0 + 3, p0 + 6, "gravity in base frame"),
        "dof_pos": ObsFieldSpec("dof_pos", p0 + 6, p0 + 27, "dof pos minus default"),
        "dof_vel": ObsFieldSpec("dof_vel", p0 + 27, p0 + 48, "dof velocities"),
        "previous_action": ObsFieldSpec("previous_action", p0 + 48, p0 + 69, "last applied policy action"),
        "tracking_err": ObsFieldSpec("tracking_err", p0 + 69, p0 + 85, "tracking error features"),
        "upper_body": ObsFieldSpec("upper_body", p0 + 85, p0 + 95, "upper body proprio"),
    }
    n = len(G1_POLICY_DOF_NAMES)
    name_to_idx = {n_: i for i, n_ in enumerate(G1_POLICY_DOF_NAMES)}
    dv0 = fields["dof_vel"].start
    for jn in G1_POLICY_DOF_NAMES:
        if "ankle" in jn:
            i = name_to_idx[jn]
            key = jn.replace("_joint", "")
            fields[f"ankle_vel_{jn}"] = ObsFieldSpec(
                f"ankle_vel_{jn}", dv0 + i, dv0 + i + 1, f"per-joint dof_vel slot for {jn}"
            )
    fields["ankle_vel"] = ObsFieldSpec(
        "ankle_vel",
        dv0 + name_to_idx["left_ankle_pitch_joint"],
        dv0 + name_to_idx["right_ankle_roll_joint"] + 1,
        "all ankle dof_vel slots (4 dims)",
    )
    return fields


OBS_FIELD_REGISTRY = build_g1_obs_field_registry()


def _match_joint(name: str, token: str) -> bool:
    t = token.lower()
    if t in ("left_leg", "right_leg"):
        side = "left" if t.startswith("left") else "right"
        return name.startswith(side) and any(
            x in name for x in ("hip", "knee", "ankle")
        )
    if t == "legs" or t == "lower":
        return any(x in name for x in ("hip", "knee", "ankle"))
    if t == "upper":
        return any(x in name for x in ("waist", "shoulder", "elbow", "wrist"))
    if t == "waist":
        return "waist" in name
    if t == "arms":
        return any(x in name for x in ("shoulder", "elbow", "wrist"))
    if t == "hip_roll":
        return "hip_roll" in name
    if t == "hip_yaw":
        return "hip_yaw" in name
    if t == "hip_pitch":
        return "hip_pitch" in name
    if t == "knee":
        return "knee" in name
    if t == "ankle":
        return "ankle" in name
    if t == "ankle_pitch":
        return "ankle_pitch" in name
    if t == "ankle_roll":
        return "ankle_roll" in name
    if t == "left_ankle":
        return name.startswith("left_") and "ankle" in name
    if t == "right_ankle":
        return name.startswith("right_") and "ankle" in name
    if t == "left_hip_roll":
        return name == "left_hip_roll_joint"
    if t == "right_hip_roll":
        return name == "right_hip_roll_joint"
    if t.endswith("_joint") and name == t:
        return True
    return False


def build_joint_group_indices(
    dof_names: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    names = tuple(dof_names or G1_POLICY_DOF_NAMES)
    known_groups = (
        "lower", "legs", "upper", "waist", "arms",
        "hip_roll", "hip_yaw", "hip_pitch", "knee",
        "ankle", "ankle_pitch", "ankle_roll",
        "left_leg", "right_leg", "left_ankle", "right_ankle",
        "left_hip_roll", "right_hip_roll",
    )
    out: Dict[str, np.ndarray] = {}
    for g in known_groups:
        idx = [i for i, n in enumerate(names) if _match_joint(n, g)]
        if not idx:
            raise ValueError(f"joint group '{g}' has no members in dof_names={names}")
        out[g] = np.asarray(idx, dtype=np.int32)
    if "legs" in out:
        out["lower"] = out["legs"]
    return out


def parse_ablate_action_groups(spec: str | None) -> Dict[str, float]:
    if not spec or spec.strip().lower() in ("none", ""):
        return {}
    scales: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid ablate-action-groups token '{part}', expected group=scale")
        g, v = part.split("=", 1)
        g = g.strip()
        scales[g] = float(v.strip())
    return scales


def apply_group_action_scale(
    action: np.ndarray,
    group_scales: Mapping[str, float],
    group_indices: Mapping[str, np.ndarray],
) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    for g, scale in group_scales.items():
        if g not in group_indices:
            raise ValueError(
                f"unknown action group '{g}'; known: {sorted(group_indices.keys())}"
            )
        out[group_indices[g]] *= float(scale)
    return out


def parse_obs_ablation(spec: str | None) -> List[Tuple[str, str, Optional[float]]]:
    """Return list of (field, mode, param)."""
    if not spec or spec.strip().lower() in ("none", ""):
        return []
    rules: List[Tuple[str, str, Optional[float]]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid obs-ablation token '{part}'")
        field, mode = part.split("=", 1)
        field, mode = field.strip(), mode.strip()
        param: Optional[float] = None
        if mode.startswith("lowpass:"):
            param = float(mode.split(":", 1)[1])
            mode = "lowpass"
        rules.append((field, mode, param))
        if field not in OBS_FIELD_REGISTRY and field != "history":
            raise ValueError(
                f"unknown obs field '{field}'; known: {sorted(OBS_FIELD_REGISTRY.keys()) + ['history']}"
            )
    return rules


@dataclass
class ControlSweepConfig:
    policy_action_scale_mult: float = 1.0
    ablate_action_groups: str = ""
    obs_ablation: str = ""
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    ankle_kp_scale: float = 1.0
    ankle_kd_scale: float = 1.0
    hip_roll_kp_scale: float = 1.0
    hip_roll_kd_scale: float = 1.0
    torque_limit_scale: float = 1.0
    motor_tau_s: float = 0.0
    record_control_trace: Optional[str] = None
    replay_target_trace: Optional[str] = None
    replay_torque_trace: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "policy_action_scale_mult": self.policy_action_scale_mult,
            "ablate_action_groups": self.ablate_action_groups or "none",
            "obs_ablation": self.obs_ablation or "none",
            "kp_scale": self.kp_scale,
            "kd_scale": self.kd_scale,
            "ankle_kp_scale": self.ankle_kp_scale,
            "ankle_kd_scale": self.ankle_kd_scale,
            "hip_roll_kp_scale": self.hip_roll_kp_scale,
            "hip_roll_kd_scale": self.hip_roll_kd_scale,
            "torque_limit_scale": self.torque_limit_scale,
            "motor_tau_s": self.motor_tau_s,
            "record_control_trace": self.record_control_trace,
            "replay_target_trace": self.replay_target_trace,
            "replay_torque_trace": self.replay_torque_trace,
        }

    @property
    def is_replay_mode(self) -> bool:
        return bool(self.replay_target_trace or self.replay_torque_trace)


class ObsAblationState:
    def __init__(self, rules: List[Tuple[str, str, Optional[float]]], num_actions: int):
        self.rules = rules
        self.num_actions = num_actions
        self._lp_state: Dict[str, np.ndarray] = {}
        self._raw_action_last: Optional[np.ndarray] = None

    def apply_to_single_obs(
        self,
        student_obs: np.ndarray,
        *,
        applied_action: np.ndarray,
        raw_action: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        obs = np.asarray(student_obs, dtype=np.float32).reshape(-1).copy()
        diffs: Dict[str, float] = {}
        for field, mode, param in self.rules:
            if field == "history":
                continue
            if field == "ankle_vel":
                spec = OBS_FIELD_REGISTRY["ankle_vel"]
            elif field in OBS_FIELD_REGISTRY:
                spec = OBS_FIELD_REGISTRY[field]
            else:
                raise ValueError(f"unknown obs field {field}")
            seg = obs[spec.start:spec.end]
            before_rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
            if field == "ankle_vel" or field.startswith("ankle_vel_"):
                if mode == "zero":
                    obs[spec.start:spec.end] = 0.0
                elif mode == "raw":
                    pass
                else:
                    raise ValueError(f"ankle_vel mode must be zero|raw, got {mode}")
            elif field == "previous_action":
                if mode == "applied_action":
                    obs[spec.start:spec.end] = applied_action
                elif mode == "raw_action":
                    if raw_action is None:
                        raise ValueError("previous_action=raw_action requires raw_action")
                    obs[spec.start:spec.end] = raw_action
                elif mode == "zero":
                    obs[spec.start:spec.end] = 0.0
                else:
                    raise ValueError(f"previous_action mode must be applied_action|raw_action|zero")
            elif mode == "lowpass":
                alpha = float(param if param is not None else 0.3)
                key = field
                prev = self._lp_state.get(key, seg.copy())
                filtered = alpha * seg + (1.0 - alpha) * prev
                self._lp_state[key] = filtered.copy()
                obs[spec.start:spec.end] = filtered
            else:
                raise ValueError(f"unsupported obs ablation {field}={mode}")
            after = obs[spec.start:spec.end]
            diffs[field] = before_rms - float(np.sqrt(np.mean(after * after))) if after.size else 0.0
        return obs, diffs

    def apply_history_ablation(self, history_buf: np.ndarray, current: np.ndarray) -> np.ndarray:
        for field, mode, _ in self.rules:
            if field == "history" and mode == "repeat_current":
                n = history_buf.shape[0]
                return np.tile(current.reshape(1, -1), (n, 1))
        return history_buf


class ControlTraceRecorder:
    def __init__(self, path: str, num_actions: int):
        self.path = path
        self.num_actions = num_actions
        self._buf: Dict[str, list] = {k: [] for k in self._keys()}

    @staticmethod
    def _keys() -> Tuple[str, ...]:
        return (
            "time", "ref_time", "raw_action", "scaled_action", "group_ablated_action",
            "applied_action", "ref_dof_pos", "target_raw", "target_after_offset_clip",
            "target_after_joint_limit", "qpos", "qvel", "dof_pos", "dof_vel",
            "torque_raw_pre_clip", "torque_post_clip", "torque_after_motor_filter",
            "contact_flags", "foot_normal_force", "foot_tangential_force",
            "foot_pos_world", "foot_vel_world", "base_pos", "base_quat",
            "base_lin_vel", "base_ang_vel", "root_xy_err", "yaw_err", "yaw_drift",
            "stance_slip_speed", "contact_switch_flags",
            "obs_ablation_rms_diff",
        )

    def append(self, **kwargs: Any) -> None:
        for k in self._keys():
            if k in kwargs:
                self._buf[k].append(kwargs[k])

    def save(self) -> str:
        out: Dict[str, Any] = {}
        for k, v in self._buf.items():
            if not v:
                continue
            if k == "obs_ablation_rms_diff":
                out[k] = v
                continue
            try:
                out[k] = np.stack(v, axis=0)
            except ValueError:
                out[k] = np.asarray(v, dtype=object)
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **out)
        return str(path)


def load_control_trace(path: str) -> Dict[str, np.ndarray]:
    data = dict(np.load(path, allow_pickle=True))
    return data


def _hf_rms(x: np.ndarray, fs: float, cutoff: float = 25.0) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 4:
        return 0.0
    try:
        from scipy import signal  # type: ignore
        nyq = 0.5 * fs
        wn = min(0.99, max(1e-4, cutoff / nyq))
        b, a = signal.butter(2, wn, btype="low")
        low = signal.filtfilt(b, a, x)
        hf = x - low
    except Exception:
        alpha = 1.0 / (1.0 + 2.0 * np.pi * cutoff * (1.0 / max(fs, 1e-6)))
        low = np.empty_like(x)
        low[0] = x[0]
        for i in range(1, x.size):
            low[i] = alpha * x[i] + (1 - alpha) * low[i - 1]
        hf = x - low
    return float(np.sqrt(np.mean(hf * hf)))


def _norm_series(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim == 1:
        return np.array([np.linalg.norm(a)])
    return np.linalg.norm(a, axis=1)


def build_enhanced_summary(
    jitter_summary: Dict[str, Any],
    ablation_meta: Dict[str, Any],
    *,
    motion_name: str,
    checkpoint_step: Optional[str],
    termination_reason: str,
    control_dt: float,
    group_indices: Mapping[str, np.ndarray],
    trace_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    core = jitter_summary
    sim2 = core.get("sim2sim_summary") or {}
    per_foot = core.get("per_foot") or []
    contact_hz = float(np.mean([f.get("contact_toggle_rate_hz", 0.0) for f in per_foot])) if per_foot else 0.0
    stance_slip_hf = float(np.mean([f.get("foot_slip_speed_hf_rms_m_s", 0.0) for f in per_foot])) if per_foot else 0.0
    stance_slip_mean = float(np.mean([f.get("mean_slip_speed_m_s", 0.0) for f in per_foot])) if per_foot else 0.0
    slip_dist = float(np.sum([f.get("integrated_slip_distance_m", 0.0) for f in per_foot])) if per_foot else 0.0

    raw_hf = core.get("policy_action_norm_hf_rms", 0.0)
    ctrl_hf = core.get("ctrl_norm_hf_rms", 0.0)
    applied_hf = raw_hf
    torque_hf = 0.0
    target_rate = 0.0
    if trace_arrays:
        if "raw_action" in trace_arrays:
            raw_hf = _hf_rms(_norm_series(trace_arrays["raw_action"]), 1.0 / max(control_dt, 1e-6))
        if "applied_action" in trace_arrays:
            applied_hf = _hf_rms(_norm_series(trace_arrays["applied_action"]), 1.0 / max(control_dt, 1e-6))
        if "torque_after_motor_filter" in trace_arrays:
            t = trace_arrays["torque_after_motor_filter"]
            torque_hf = _hf_rms(np.linalg.norm(t, axis=1), 1.0 / max(control_dt, 1e-6))
        if "target_after_joint_limit" in trace_arrays:
            tgt = trace_arrays["target_after_joint_limit"]
            rate = np.diff(tgt, axis=0) / max(control_dt, 1e-6)
            target_rate = float(np.sqrt(np.mean(rate * rate))) if rate.size else 0.0

    vr = (core.get("sim2sim_motion_phase") or {}).get("valid_reference_segment") or {}
    yaw_drift = vr.get("yaw_drift")
    if yaw_drift is None:
        yaw_drift = (core.get("motion_phase_quick_view") or {}).get("valid_yaw_drift_mean_rad")

    out: Dict[str, Any] = {
        "motion_name": motion_name,
        "checkpoint_step": checkpoint_step,
        "sim_duration": ablation_meta.get("sim_duration"),
        "actual_duration": core.get("duration_s"),
        "termination_reason": termination_reason,
        **{k: ablation_meta.get(k) for k in (
            "policy_action_scale_mult", "ablate_action_groups", "obs_ablation",
            "kp_scale", "kd_scale", "ankle_kd_scale", "hip_roll_kd_scale",
            "torque_limit_scale", "motor_tau_s", "no_action_delay", "no_action_lowpass",
            "action_filter_alpha", "v4", "g1_balance_safe_enable",
        )},
        "vel_xy_rmse": sim2.get("vel_xy_rmse_m_s"),
        "root_xy_err_mean": vr.get("root_xy_err_mean"),
        "root_xy_err_final": vr.get("root_xy_err_final"),
        "yaw_drift_abs": abs(float(yaw_drift)) if yaw_drift is not None else None,
        "yaw_err_rmse": vr.get("yaw_err_rmse"),
        "base_roll_max": vr.get("base_roll_max"),
        "base_pitch_max": vr.get("base_pitch_max"),
        "base_lin_vel_xy_mean": vr.get("mean_base_xy_speed"),
        "base_lin_vel_xy_rmse_on_stationary_ref": sim2.get("stationary_mean_base_xy_speed_m_s"),
        "left_contact_rate": per_foot[0].get("contact_toggle_rate_hz") if len(per_foot) > 0 else None,
        "right_contact_rate": per_foot[1].get("contact_toggle_rate_hz") if len(per_foot) > 1 else None,
        "contact_switch_hz_mean": contact_hz,
        "stance_slip_speed_mean": stance_slip_mean,
        "stance_slip_speed_hf_rms": stance_slip_hf,
        "stance_slip_distance_total": slip_dist,
        "foot_normal_force_hf_rms": float(np.mean([f.get("normal_force_hf_rms_n", 0) for f in per_foot])) if per_foot else 0.0,
        "foot_tangential_force_hf_rms": float(np.mean([f.get("tangential_force_hf_rms_n", 0) for f in per_foot])) if per_foot else 0.0,
        "raw_action_hf_rms": raw_hf,
        "applied_action_hf_rms": applied_hf,
        "target_rate_rms": target_rate,
        "torque_hf_rms": torque_hf,
        "ctrl_hf_rms": ctrl_hf,
        "diagnosis_tags": [],
        "ranked_suspected_causes": [],
    }

    tags, ranked = _rank_diagnosis(out, core.get("diagnosis", ""))
    out["diagnosis_tags"] = tags
    out["ranked_suspected_causes"] = ranked

    if trace_arrays and group_indices:
        out["action_rms_by_group"] = _group_stats(trace_arrays.get("applied_action"), group_indices, hf=False)
        out["action_hf_rms_by_group"] = _group_stats(trace_arrays.get("applied_action"), group_indices, hf=True, dt=control_dt)

    return out


def _group_stats(
    actions: Optional[np.ndarray],
    group_indices: Mapping[str, np.ndarray],
    *,
    hf: bool,
    dt: float = 0.02,
) -> Dict[str, float]:
    if actions is None:
        return {}
    a = np.asarray(actions, dtype=np.float64)
    out: Dict[str, float] = {}
    fs = 1.0 / max(dt, 1e-6)
    for g, idx in group_indices.items():
        if g in ("lower", "legs"):
            continue
        seg = a[:, idx]
        n = np.linalg.norm(seg, axis=1)
        if hf:
            out[g] = _hf_rms(n, fs)
        else:
            out[g] = float(np.sqrt(np.mean(n * n)))
    return out


def _rank_diagnosis(metrics: Dict[str, Any], base_diagnosis: str) -> Tuple[List[str], List[str]]:
    tags: List[str] = []
    if base_diagnosis:
        tags.append(base_diagnosis)
    scores: List[Tuple[float, str]] = []

    ctrl_hf = float(metrics.get("ctrl_hf_rms") or 0.0)
    raw_hf = float(metrics.get("raw_action_hf_rms") or 0.0)
    slip_hf = float(metrics.get("stance_slip_speed_hf_rms") or 0.0)
    contact_hz = float(metrics.get("contact_switch_hz_mean") or 0.0)
    yaw = float(metrics.get("yaw_drift_abs") or 0.0)
    policy_scale = float(metrics.get("policy_action_scale_mult") or 1.0)

    if ctrl_hf > 0.5 and raw_hf < 0.15:
        t = "likely_high_frequency_jitter_induced_slip"
        if t not in tags:
            tags.append(t)
        scores.append((ctrl_hf, t))

    if policy_scale >= 0.99 and slip_hf > 0.05 and raw_hf > 0.08:
        t = "likely_residual_too_large"
        if t not in tags:
            tags.append(t)
        scores.append((raw_hf, t))

    if contact_hz > 15 and slip_hf > 0.03:
        t = "likely_ankle_pd_contact_chatter"
        if t not in tags:
            tags.append(t)
        scores.append((contact_hz * slip_hf, t))

    if abs(float(metrics.get("root_xy_err_mean") or 0.0)) > 0.3:
        t = "likely_hip_roll_lateral_bias"
        if t not in tags:
            tags.append(t)
        scores.append((abs(float(metrics.get("root_xy_err_mean") or 0.0)), t))

    if yaw > 1.0:
        t = "likely_upper_body_yaw_momentum_failure"
        if t not in tags:
            tags.append(t)
        scores.append((yaw, t))

    if ctrl_hf > 0.4 and raw_hf < 0.1:
        t = "likely_obs_feedback_phase_issue"
        if t not in tags:
            tags.append(t)
        scores.append((ctrl_hf - raw_hf, t))

    if slip_hf > 0.08 and ctrl_hf > 0.6:
        t = "likely_torque_contact_instability"
        if t not in tags:
            tags.append(t)
        scores.append((slip_hf * ctrl_hf, t))

    scores.sort(key=lambda x: -x[0])
    ranked = [s[1] for s in scores]
    return tags, ranked


def format_run_config_dir_name(cfg: ControlSweepConfig, default: str = "baseline") -> str:
    if (
        cfg.policy_action_scale_mult == 1.0
        and not cfg.ablate_action_groups
        and not cfg.obs_ablation
        and cfg.kp_scale == 1.0
        and cfg.kd_scale == 1.0
        and cfg.ankle_kd_scale == 1.0
        and cfg.hip_roll_kd_scale == 1.0
        and cfg.motor_tau_s == 0.0
        and cfg.torque_limit_scale == 1.0
    ):
        return default
    parts: List[str] = []
    if cfg.policy_action_scale_mult != 1.0:
        parts.append(f"action_scale_{cfg.policy_action_scale_mult:.2f}".replace(".", "p"))
    if cfg.ablate_action_groups:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", cfg.ablate_action_groups)[:48]
        parts.append(f"ablate_{safe}")
    if cfg.obs_ablation:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", cfg.obs_ablation)[:48]
        parts.append(f"obs_{safe}")
    if cfg.kd_scale != 1.0:
        parts.append(f"kd_{cfg.kd_scale:.2f}".replace(".", "p"))
    if cfg.kp_scale != 1.0:
        parts.append(f"kp_{cfg.kp_scale:.2f}".replace(".", "p"))
    if cfg.ankle_kd_scale != 1.0:
        parts.append(f"ankle_kd_{cfg.ankle_kd_scale:.2f}".replace(".", "p"))
    if cfg.hip_roll_kd_scale != 1.0:
        parts.append(f"hip_roll_kd_{cfg.hip_roll_kd_scale:.2f}".replace(".", "p"))
    if cfg.motor_tau_s > 0:
        parts.append(f"motor_tau_{cfg.motor_tau_s:.2f}".replace(".", "p"))
    if cfg.torque_limit_scale != 1.0:
        parts.append(f"torque_lim_{cfg.torque_limit_scale:.2f}".replace(".", "p"))
    return "_".join(parts) if parts else default


SWEEP_PRESETS: Dict[str, List[ControlSweepConfig]] = {
    "none": [ControlSweepConfig()],
    "a1_contact_debug": [
        ControlSweepConfig(),
        ControlSweepConfig(policy_action_scale_mult=0.5),
        ControlSweepConfig(ablate_action_groups="ankle=0.5"),
        ControlSweepConfig(ablate_action_groups="ankle_roll=0.0"),
        ControlSweepConfig(ablate_action_groups="hip_roll=0.5"),
        ControlSweepConfig(kd_scale=0.5),
        ControlSweepConfig(ankle_kd_scale=0.5),
        ControlSweepConfig(hip_roll_kd_scale=0.5),
        ControlSweepConfig(motor_tau_s=0.02),
        ControlSweepConfig(motor_tau_s=0.04),
        ControlSweepConfig(obs_ablation="base_ang_vel=lowpass:0.3"),
        ControlSweepConfig(obs_ablation="previous_action=zero"),
    ],
    "a3_upper_debug": [
        ControlSweepConfig(),
        ControlSweepConfig(ablate_action_groups="upper=0.0"),
        ControlSweepConfig(ablate_action_groups="arms=0.0"),
        ControlSweepConfig(ablate_action_groups="waist=0.0"),
        ControlSweepConfig(ablate_action_groups="ankle=0.5"),
        ControlSweepConfig(policy_action_scale_mult=0.5),
    ],
    "throwing_yaw_debug": [
        ControlSweepConfig(),
        ControlSweepConfig(ablate_action_groups="upper=0.0"),
        ControlSweepConfig(ablate_action_groups="arms=0.0"),
        ControlSweepConfig(ablate_action_groups="waist=0.0"),
        ControlSweepConfig(ablate_action_groups="hip_roll=0.5"),
        ControlSweepConfig(ablate_action_groups="ankle=0.5"),
        ControlSweepConfig(kd_scale=0.5),
        ControlSweepConfig(motor_tau_s=0.04),
    ],
}


def expand_sweep_configs(
    *,
    preset: str,
    policy_action_scale_mults: Optional[List[float]] = None,
    ablate_action_groups_list: Optional[List[str]] = None,
    kd_scales: Optional[List[float]] = None,
    kp_scales: Optional[List[float]] = None,
    motor_tau_s_list: Optional[List[float]] = None,
    torque_limit_scales: Optional[List[float]] = None,
    ankle_kd_scales: Optional[List[float]] = None,
    hip_roll_kd_scales: Optional[List[float]] = None,
    obs_ablations_list: Optional[List[str]] = None,
) -> List[ControlSweepConfig]:
    if preset and preset != "none":
        return list(SWEEP_PRESETS.get(preset, SWEEP_PRESETS["none"]))

    configs = [ControlSweepConfig()]
    if policy_action_scale_mults:
        configs = [ControlSweepConfig(policy_action_scale_mult=m) for m in policy_action_scale_mults]
    if ablate_action_groups_list:
        base = configs
        configs = []
        for b in base:
            for g in ablate_action_groups_list:
                if g.lower() == "none":
                    configs.append(ControlSweepConfig(**{**b.__dict__, "ablate_action_groups": ""}))
                else:
                    configs.append(ControlSweepConfig(**{**b.__dict__, "ablate_action_groups": g}))
    # Cartesian product for control scales (only if explicitly provided)
    def _expand_attr(attr: str, values: Optional[List[float]], field: str) -> None:
        nonlocal configs
        if not values:
            return
        new_configs: List[ControlSweepConfig] = []
        for b in configs:
            for v in values:
                d = dict(b.__dict__)
                d[field] = v
                new_configs.append(ControlSweepConfig(**d))
        configs = new_configs

    _expand_attr("kd_scale", kd_scales, "kd_scale")
    _expand_attr("kp_scale", kp_scales, "kp_scale")
    _expand_attr("motor_tau_s", motor_tau_s_list, "motor_tau_s")
    _expand_attr("torque_limit_scale", torque_limit_scales, "torque_limit_scale")
    _expand_attr("ankle_kd_scale", ankle_kd_scales, "ankle_kd_scale")
    _expand_attr("hip_roll_kd_scale", hip_roll_kd_scales, "hip_roll_kd_scale")
    if obs_ablations_list:
        base = configs
        configs = []
        for b in base:
            for o in obs_ablations_list:
                d = dict(b.__dict__)
                d["obs_ablation"] = "" if o.lower() == "none" else o
                configs.append(ControlSweepConfig(**d))
    return configs


def write_aggregate_ablation_summary(
    summary_paths: List[Tuple[str, str, Path]],
    output_csv: Path,
) -> None:
    """summary_paths: (motion_name, run_config_name, summary_json_path)."""
    import csv as csv_mod

    rows: List[Dict[str, Any]] = []
    for motion, run_name, path in summary_paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        enhanced = data.get("policy_in_loop_summary") or data
        rows.append({
            "motion_name": motion,
            "checkpoint_step": enhanced.get("checkpoint_step"),
            "run_config_name": run_name,
            "policy_action_scale_mult": enhanced.get("policy_action_scale_mult"),
            "ablate_action_groups": enhanced.get("ablate_action_groups"),
            "obs_ablation": enhanced.get("obs_ablation"),
            "kp_scale": enhanced.get("kp_scale"),
            "kd_scale": enhanced.get("kd_scale"),
            "ankle_kd_scale": enhanced.get("ankle_kd_scale"),
            "hip_roll_kd_scale": enhanced.get("hip_roll_kd_scale"),
            "motor_tau_s": enhanced.get("motor_tau_s"),
            "vel_xy_rmse": enhanced.get("vel_xy_rmse"),
            "root_xy_err_mean": enhanced.get("root_xy_err_mean"),
            "yaw_drift_abs": enhanced.get("yaw_drift_abs"),
            "contact_switch_hz_mean": enhanced.get("contact_switch_hz_mean"),
            "stance_slip_speed_hf_rms": enhanced.get("stance_slip_speed_hf_rms"),
            "stance_slip_speed_mean": enhanced.get("stance_slip_speed_mean"),
            "stance_slip_distance_total": enhanced.get("stance_slip_distance_total"),
            "ctrl_hf_rms": enhanced.get("ctrl_hf_rms"),
            "target_rate_rms": enhanced.get("target_rate_rms"),
            "torque_rate_rms": enhanced.get("torque_rate_rms"),
            "raw_action_hf_rms": enhanced.get("raw_action_hf_rms"),
            "applied_action_hf_rms": enhanced.get("applied_action_hf_rms"),
            "diagnosis_tags": ";".join(enhanced.get("diagnosis_tags") or []),
            "ranked_suspected_causes": ";".join(enhanced.get("ranked_suspected_causes") or []),
        })

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv_mod.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Best configs per motion
    by_motion: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_motion.setdefault(str(r["motion_name"]), []).append(r)

    def _norm(vals: List[float]) -> List[float]:
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [0.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    best_lines: List[str] = []
    for motion, mrows in by_motion.items():
        slips = [float(r.get("stance_slip_speed_hf_rms") or 0) for r in mrows]
        ctrls = [float(r.get("ctrl_hf_rms") or 0) for r in mrows]
        vels = [float(r.get("vel_xy_rmse") or 0) for r in mrows]
        roots = [float(r.get("root_xy_err_mean") or 0) for r in mrows]
        yaws = [float(r.get("yaw_drift_abs") or 0) for r in mrows]
        contacts = [float(r.get("contact_switch_hz_mean") or 0) for r in mrows]
        ns = _norm(slips)
        nc = _norm(ctrls)
        nv = _norm(vels)
        nr = _norm(roots)
        ny = _norm(yaws)
        nct = _norm(contacts)
        scores = [
            1.0 * nv[i] + 1.0 * nr[i] + 1.0 * ny[i] + 2.0 * ns[i] + 1.5 * nct[i] + 1.0 * nc[i]
            for i in range(len(mrows))
        ]
        i_slip = int(np.argmin(slips)) if slips else 0
        i_ctrl = int(np.argmin(ctrls)) if ctrls else 0
        i_vel = int(np.argmin(vels)) if vels else 0
        i_trade = int(np.argmin(scores)) if scores else 0
        best_lines.append(
            f"{motion}: best_by_low_slip={mrows[i_slip]['run_config_name']}, "
            f"best_by_low_ctrl_hf={mrows[i_ctrl]['run_config_name']}, "
            f"best_by_low_vel_rmse={mrows[i_vel]['run_config_name']}, "
            f"best_tradeoff_score={mrows[i_trade]['run_config_name']}"
        )

    with open(output_csv.with_suffix(".best.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(best_lines) + "\n")
