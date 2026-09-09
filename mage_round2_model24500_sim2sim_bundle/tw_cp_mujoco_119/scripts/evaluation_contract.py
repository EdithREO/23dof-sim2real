"""Small, dependency-free helpers for fail-closed sim2sim evaluation contracts."""

from __future__ import annotations

import hashlib
import math
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Mapping


CONTRACT_SCHEMA = "g1-mage-sim2sim-contract/v2"
BASELINE_DELAY_STEPS = 1
BASELINE_ACTION_FILTER_ALPHA = 0.5
BASELINE_MOTOR_TAU_S = 0.0
POST_MOTION_HOLD_S = 10.0
CONTROL_DELAY_CANDIDATES = (0, 1, 2)
CONTROL_ALPHA_CANDIDATES = (0.4, 0.5, 0.6)

# This is the single source of truth for the MAGE80250 G1 deployment ABI.
# Keep it JSON-shaped and dependency-free: the evaluator, runner, and CPU-only
# contract checks all consume this object rather than spelling out copies of
# the training/deployment contract.
CANONICAL_ABI_SCHEMA = "g1-mage-canonical-abi/v1"
CANONICAL_POLICY_JOINT_ORDER = (
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
CANONICAL_ACTION_SCALE = (
    0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
    0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
    0.547546,
    0.438577, 0.438577, 0.438577, 0.438577,
    0.438577, 0.438577, 0.438577, 0.438577,
)
CANONICAL_KP = (
    40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
    40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
    40.179238,
    14.250623, 14.250623, 14.250623, 14.250623,
    14.250623, 14.250623, 14.250623, 14.250623,
)
CANONICAL_KD = (
    2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
    2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
    2.557890,
    0.907223, 0.907223, 0.907223, 0.907223,
    0.907223, 0.907223, 0.907223, 0.907223,
)
CANONICAL_EFFORT_LIMITS = (
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
)
CANONICAL_VELOCITY_LIMITS = (
    32.0, 20.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 20.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0,
)

# The 106-value student327 frame is part of the model ABI, not merely a
# diagnostic label list.  Keep names and scales here so a deploy-side runner
# cannot silently reorder or rescale one field while preserving 1166 inputs.
CANONICAL_MIMIC_OBSERVATION_FIELDS = (
    "mimic/root_height",
    "mimic/ref_roll",
    "mimic/ref_pitch",
    "mimic/ref_heading_sin",
    "mimic/ref_heading_cos",
    "mimic/ref_vel_x",
    "mimic/ref_vel_y",
    "mimic/ref_vel_z",
    "mimic/ref_ang_vel_yaw",
) + tuple(f"mimic/ref_dof_pos_{index}" for index in range(21)) + (
    "mimic/future_5/ref_vel_x",
    "mimic/future_5/ref_vel_y",
    "mimic/future_5/ref_vel_z",
    "mimic/future_5/ref_heading_sin",
    "mimic/future_5/ref_heading_cos",
    "mimic/future_5/ref_ang_vel_yaw",
)
CANONICAL_PROPRIO_OBSERVATION_FIELDS = (
    "proprio/base_ang_vel_x",
    "proprio/base_ang_vel_y",
    "proprio/base_ang_vel_z",
    "proprio/roll",
    "proprio/pitch",
    "proprio/yaw_sin",
    "proprio/yaw_cos",
) + tuple(
    f"proprio/dof_pos_minus_deploy_default_{index}" for index in range(21)
) + tuple(f"proprio/dof_vel_{index}" for index in range(21)) + tuple(
    f"proprio/deploy_action_history_{index}" for index in range(21)
)
CANONICAL_OBSERVATION_FIELDS = (
    CANONICAL_MIMIC_OBSERVATION_FIELDS + CANONICAL_PROPRIO_OBSERVATION_FIELDS
)
CANONICAL_MIMIC_OBSERVATION_SCALES = (1.0,) * len(CANONICAL_MIMIC_OBSERVATION_FIELDS)
CANONICAL_PROPRIO_OBSERVATION_SCALES = (
    (0.25,) * 3
    + (1.0,) * 4
    + (1.0,) * 21
    + (0.05,) * 21
    + (1.0,) * 21
)
CANONICAL_OBSERVATION_SCALES = (
    CANONICAL_MIMIC_OBSERVATION_SCALES + CANONICAL_PROPRIO_OBSERVATION_SCALES
)
CANONICAL_ANKLE_VELOCITY_MASK_INDICES = (4, 5, 10, 11)


def _observation_segment(
    name: str,
    offset: int,
    fields: tuple[str, ...],
    scales: tuple[float, ...],
) -> dict[str, object]:
    return {
        "name": name,
        "offset": int(offset),
        "dim": len(fields),
        "fields": list(fields),
        "scales": list(scales),
    }

CANONICAL_ABI: dict[str, object] = {
    "schema": CANONICAL_ABI_SCHEMA,
    "policy_io": {
        "input_dim": 1166,
        "output_dim": 21,
        "input_shape": ["batch", 1166],
        "output_shape": ["batch", 21],
        "history_length": 10,
        "observation_single_dim": 106,
        "layout": "current_then_history_oldest_to_newest",
        "input_dim_formula": "current_106 + history_10x106 = 1166",
    },
    "observation": {
        "single": {
            "dim": 106,
            "segments": [
                _observation_segment(
                    "mimic",
                    0,
                    CANONICAL_MIMIC_OBSERVATION_FIELDS,
                    CANONICAL_MIMIC_OBSERVATION_SCALES,
                ),
                _observation_segment(
                    "proprio",
                    36,
                    CANONICAL_PROPRIO_OBSERVATION_FIELDS,
                    CANONICAL_PROPRIO_OBSERVATION_SCALES,
                ),
            ],
            "field_names": list(CANONICAL_OBSERVATION_FIELDS),
            "scales": list(CANONICAL_OBSERVATION_SCALES),
            "ankle_velocity_mask": {
                "dof_indices": list(CANONICAL_ANKLE_VELOCITY_MASK_INDICES),
                "field_prefix": "proprio/dof_vel_",
                "output_value": 0.0,
                "reason": "parallel_ankle_joint_velocity_omitted",
            },
        },
        "history": {
            "frame_count": 10,
            "frame_dim": 106,
            "flatten_order": "oldest_to_newest",
            "policy_input_order": ["current", "history"],
            "buffer_reset": "zeros",
            "first_observation_fill": "current_frame",
            "update": "drop_oldest_append_current",
        },
        "last_action": {
            "field_prefix": "proprio/deploy_action_history_",
            "dim": 21,
            "scale": 1.0,
            "source": "filtered_clipped_delayed_applied_action",
            "source_stage": "postprocess_policy_action_return_before_pd",
            "history_projection": "current_frame_then_previous_two_frames",
            "history_steps": 3,
            "prior_frame_count": 2,
            "reset_current": "zeros",
            "reset_previous_two": "zeros_until_actions_are_applied",
        },
    },
    "control_rate_hz": 50.0,
    "base_ang_vel": {
        "observation_frame": "local",
        "reset_source_frame": "world",
        "reset_target_frame": "local",
        "conversion": "world_to_local_quaternion_conjugate",
    },
    "reset": {
        "root_z_offset_m": 0.05,
        "angular_velocity_source_frame": "world",
        "angular_velocity_target_frame": "local",
    },
    "action": {
        "mode": "residual_ref",
        "clip_rad": 0.6,
        "scale": list(CANONICAL_ACTION_SCALE),
    },
    "pd": {
        "kp": list(CANONICAL_KP),
        "kd": list(CANONICAL_KD),
    },
    "limits": {
        "effort": list(CANONICAL_EFFORT_LIMITS),
        "velocity": list(CANONICAL_VELOCITY_LIMITS),
        "unit_effort": "N*m",
        "unit_velocity": "rad/s",
    },
    "joint_order": list(CANONICAL_POLICY_JOINT_ORDER),
    "xml_topology": {
        "wrist_roll_mode": "xml_topology_fixed_omitted_from_21_actions",
        "fixed_wrist_roll_joints": ["left_wrist_roll_joint", "right_wrist_roll_joint"],
        "policy_joint_count": 21,
    },
    "control_contract": {
        "baseline": {
            "delay_steps": BASELINE_DELAY_STEPS,
            "action_filter_alpha": BASELINE_ACTION_FILTER_ALPHA,
            "motor_tau_s": BASELINE_MOTOR_TAU_S,
        },
        "sweep": {
            "delay_steps": list(CONTROL_DELAY_CANDIDATES),
            "action_filter_alpha": list(CONTROL_ALPHA_CANDIDATES),
            "motor_tau_s": BASELINE_MOTOR_TAU_S,
        },
    },
    "horizon": {
        "post_motion_hold_s": POST_MOTION_HOLD_S,
        "terminal_velocity_decay": "cosine",
        "motor_tau_s": BASELINE_MOTOR_TAU_S,
        "horizon_truncated_is_failure": False,
    },
}


def canonical_abi() -> dict[str, object]:
    """Return an independent JSON-shaped copy of the locked ABI."""
    return copy.deepcopy(CANONICAL_ABI)


def canonical_observation_abi() -> dict[str, object]:
    """Return the dependency-free observation portion of the locked ABI."""
    observation = CANONICAL_ABI["observation"]
    if not isinstance(observation, Mapping):
        raise ValueError("internal canonical observation ABI is not a mapping")
    return copy.deepcopy(dict(observation))


def validate_observation_abi(abi: Mapping[str, object]) -> dict[str, object]:
    """Validate every observation field, scale, mask, and history rule."""
    observation = abi.get("observation")
    expected = CANONICAL_ABI["observation"]
    if not isinstance(observation, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("canonical ABI is missing its observation mapping")
    differences = canonical_abi_differences(observation, expected)
    if differences:
        raise ValueError(f"canonical observation ABI mismatch at: {differences[:12]}")
    single = observation.get("single")
    if not isinstance(single, Mapping):
        raise ValueError("canonical observation ABI is missing single-frame layout")
    if int(single.get("dim", -1)) != len(CANONICAL_OBSERVATION_FIELDS):
        raise ValueError("canonical observation dimension does not match field count")
    if tuple(single.get("field_names", ())) != CANONICAL_OBSERVATION_FIELDS:
        raise ValueError("canonical observation field order is not frozen")
    if tuple(float(value) for value in single.get("scales", ())) != CANONICAL_OBSERVATION_SCALES:
        raise ValueError("canonical observation scales are not frozen")
    segments = single.get("segments")
    if not isinstance(segments, list) or len(segments) != 2:
        raise ValueError("canonical observation must contain mimic and proprio segments")
    for segment, name, offset, fields, scales in (
        (segments[0], "mimic", 0, CANONICAL_MIMIC_OBSERVATION_FIELDS, CANONICAL_MIMIC_OBSERVATION_SCALES),
        (segments[1], "proprio", 36, CANONICAL_PROPRIO_OBSERVATION_FIELDS, CANONICAL_PROPRIO_OBSERVATION_SCALES),
    ):
        if not isinstance(segment, Mapping):
            raise ValueError(f"canonical {name} segment is not a mapping")
        if (
            segment.get("name") != name
            or int(segment.get("offset", -1)) != offset
            or int(segment.get("dim", -1)) != len(fields)
            or tuple(segment.get("fields", ())) != fields
            or tuple(float(value) for value in segment.get("scales", ())) != scales
        ):
            raise ValueError(f"canonical {name} segment differs from the locked layout")
    mask = single.get("ankle_velocity_mask")
    if not isinstance(mask, Mapping) or tuple(mask.get("dof_indices", ())) != CANONICAL_ANKLE_VELOCITY_MASK_INDICES:
        raise ValueError("canonical ankle velocity mask differs from the locked layout")
    return canonical_observation_abi()


def validate_canonical_observation_layout(
    labels: Iterable[str],
    scales: Iterable[float],
    *,
    single_dim: int,
    history_len: int,
    policy_dim: int,
    ankle_velocity_indices: Iterable[int],
) -> dict[str, object]:
    """Static runner-side check against the exact 106/1166 observation ABI."""
    label_values = tuple(str(value) for value in labels)
    scale_values = tuple(float(value) for value in scales)
    ankle_values = tuple(int(value) for value in ankle_velocity_indices)
    if label_values != CANONICAL_OBSERVATION_FIELDS:
        raise ValueError("runner observation field order differs from canonical ABI")
    if scale_values != CANONICAL_OBSERVATION_SCALES:
        raise ValueError("runner observation scales differ from canonical ABI")
    if int(single_dim) != 106 or int(history_len) != 10 or int(policy_dim) != 1166:
        raise ValueError("runner observation dimensions differ from canonical ABI")
    if ankle_values != CANONICAL_ANKLE_VELOCITY_MASK_INDICES:
        raise ValueError("runner ankle velocity mask differs from canonical ABI")
    return canonical_observation_abi()


def canonical_abi_differences(
    left: object,
    right: object,
    *,
    ignored_paths: Iterable[str] = (),
) -> list[str]:
    """Return deterministic leaf paths whose values differ."""
    ignored = {str(path) for path in ignored_paths}
    differences: list[str] = []

    def visit(a: object, b: object, path: str) -> None:
        if path in ignored or any(path.startswith(prefix + ".") for prefix in ignored):
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            keys = sorted(set(a) | set(b), key=str)
            for key in keys:
                child = f"{path}.{key}" if path else str(key)
                if key not in a or key not in b:
                    differences.append(child)
                else:
                    visit(a[key], b[key], child)
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                differences.append(path)
                return
            for index, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, f"{path}[{index}]")
            return
        if a != b:
            differences.append(path)

    visit(left, right, "")
    return differences


def validate_canonical_abi(abi: Mapping[str, object]) -> dict[str, object]:
    """Fail closed unless *abi* exactly equals this module's canonical ABI."""
    if not isinstance(abi, Mapping):
        raise ValueError("canonical ABI must be a mapping")
    validate_observation_abi(abi)
    differences = canonical_abi_differences(abi, CANONICAL_ABI)
    if differences:
        raise ValueError(f"canonical ABI mismatch at: {differences[:12]}")
    return canonical_abi()


def inspect_xml_topology(xml_path: str | Path) -> dict[str, object]:
    """Inspect fixed wrist topology without importing MuJoCo."""
    path = Path(xml_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"invalid MJCF XML: {path}") from exc
    joint_names = sorted(
        str(node.attrib["name"])
        for node in root.iter("joint")
        if isinstance(node.attrib.get("name"), str)
    )
    fixed = [name for name in ("left_wrist_roll_joint", "right_wrist_roll_joint") if name not in joint_names]
    if len(fixed) != 2:
        raise ValueError("canonical XML must omit both wrist-roll joints from the actuated topology")
    policy_names = [name for name in joint_names if name in CANONICAL_POLICY_JOINT_ORDER]
    if tuple(policy_names) != tuple(sorted(CANONICAL_POLICY_JOINT_ORDER)):
        raise ValueError("canonical XML is missing one or more policy joints")
    return {
        "wrist_roll_mode": "xml_topology_fixed_omitted_from_21_actions",
        "fixed_wrist_roll_joints": fixed,
        "joint_count_in_policy": len(policy_names),
        "joint_names_present": joint_names,
    }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streamed SHA256 for an explicitly selected regular file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path, *, role: str) -> dict[str, object]:
    file_path = Path(path).resolve()
    stat = file_path.stat()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return {
        "role": role,
        "path": str(file_path),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(file_path),
    }


def world_ang_vel_to_local_xyzw(
    quat_xyzw: Iterable[float], world_ang_vel: Iterable[float]
) -> tuple[float, float, float]:
    """Rotate a world-frame vector into the body frame for an xyzw quaternion."""
    x, y, z, w = (float(value) for value in quat_xyzw)
    vx, vy, vz = (float(value) for value in world_ang_vel)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # Rotate by the conjugate q^-1: v' = v + w*t + cross(-q_xyz, t).
    cx, cy, cz = -x, -y, -z
    tx = 2.0 * (cy * vz - cz * vy)
    ty = 2.0 * (cz * vx - cx * vz)
    tz = 2.0 * (cx * vy - cy * vx)
    return (
        vx + w * tx + (cy * tz - cz * ty),
        vy + w * ty + (cz * tx - cx * tz),
        vz + w * tz + (cx * ty - cy * tx),
    )


def cosine_terminal_velocity_scale(time_past_end_s: float, hold_s: float) -> float:
    if not math.isfinite(time_past_end_s) or not math.isfinite(hold_s):
        raise ValueError("terminal hold inputs must be finite")
    if hold_s <= 0.0:
        return 0.0 if time_past_end_s > 0.0 else 1.0
    alpha = min(max(time_past_end_s / hold_s, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * alpha))


def validate_candidate_grid(
    delays: Iterable[int], alphas: Iterable[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    delay_values = tuple(int(value) for value in delays)
    alpha_values = tuple(round(float(value), 6) for value in alphas)
    if delay_values != CONTROL_DELAY_CANDIDATES:
        raise ValueError(f"delay candidates must be {CONTROL_DELAY_CANDIDATES}")
    if alpha_values != CONTROL_ALPHA_CANDIDATES:
        raise ValueError(f"alpha candidates must be {CONTROL_ALPHA_CANDIDATES}")
    return delay_values, alpha_values


def require_finite_positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return number


def require_contract_fields(contract: Mapping[str, object], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in contract]
    if missing:
        raise ValueError(f"evaluation contract missing fields: {missing}")
