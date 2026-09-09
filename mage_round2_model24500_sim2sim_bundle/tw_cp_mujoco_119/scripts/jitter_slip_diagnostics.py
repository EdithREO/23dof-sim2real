# -*- coding: utf-8 -*-
"""
Jitter / slip diagnostics for MuJoCo sim2sim (per-physics-step logging).

用法（与 c1_mujoco_sim_cp_rmg_onnx_csv.py 配合）::

    python tw_cp_mujoco_119/scripts/c1_mujoco_sim_cp_rmg_onnx_csv.py \\
        --diagnose-jitter-slip \\
        --diagnostics-dir diagnostics \\
        --jitter-cutoff-hz 25

输出目录（默认 diagnostics/YYYYMMDD-HHMMSS/）::

    raw_timeseries.npz   # 全量时序（每步 physics step）
    summary.json         # 统计量、sim2sim 跟踪摘要、diagnosis、reasons、missing_signals
    plots.png            # 关键曲线（含速度/姿态/踝目标等，需传入 track_ctx）
    gptpro_report.md     # 运行信息与 GPTPro 阅读指引

传入 ``track_ctx``（由 sim 脚本在 ``record_after_step`` 中提供）后，将额外记录：
参考根位置/速度、机体坐标系线速度及与训练一致的 xy 速度误差、足端 body 姿态、
踝关节目标/实际角、力矩限幅前后对比等。

解读要点::

    - base_drift_xy_m: 基座水平面内相对起点的净位移（总漂移），非路径积分。
    - integrated_slip_distance_m: 接触期内 ∫ slip_speed dt，反映足端切向滑移累积。
    - *_rms / *_hf_rms: 全频段 RMS 与高通 RMS；hf 用于捕捉肉眼难辨的高频振荡。
    - diagnosis 为保守启发式结论，需结合 reasons 与曲线人工复核。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

# -----------------------------------------------------------------------------
# MuJoCo contact frame / mj_contactForce 说明（本仓库 mujoco Python 实测）：
# - data.contact[i].frame 为 9 元数组，reshape(3,3) 后 **每一行** 是世界系下接触标架的三个单位轴。
# - 第 0 行即为接触法向 n（由 geom2 指向 geom1 的约定，与 MuJoCo 文档一致）。
# - mj_contactForce 输出的 6D 量前 3 个分量位于 **接触标架** 内：
#     force6[0]  = 法向力（压为正，与上述 n 一致）
#     force6[1:3] = 两个切向分量（condim=3 时）
# 若升级 MuJoCo 后结果异常，请用单接触测试（box on plane）核对 force6 与 frame 行向量。
# -----------------------------------------------------------------------------


def _safe_json(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return obj


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def _highpass_series(x: np.ndarray, fs: float, cutoff_hz: float) -> Tuple[np.ndarray, str]:
    """返回 (high_freq_component, method_name)。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 8:
        return np.zeros_like(x), "too_short_zero_hf"

    # 优先 scipy.signal butter + filtfilt
    try:
        from scipy import signal  # type: ignore

        nyq = 0.5 * fs
        wn = min(0.99, max(1e-4, float(cutoff_hz) / nyq))
        b, a = signal.butter(2, wn, btype="low")
        if n > 3 * max(len(a), len(b)):
            low = signal.filtfilt(b, a, x)
            return x - low, "scipy_butter_filtfilt"
    except Exception:
        pass

    dt = 1.0 / max(fs, 1e-9)
    fc = max(float(cutoff_hz), 1e-3)
    tau = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (tau + dt)
    low = np.empty_like(x)
    low[0] = x[0]
    for i in range(1, n):
        low[i] = alpha * x[i] + (1.0 - alpha) * low[i - 1]
    return x - low, "ema_subtract"


def _quat_rotate_inverse_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """世界系向量 v 转到本体系；q 为 xyzw，表示 body->world。"""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    q_vec = np.array([x, y, z], dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64).reshape(3)
    a = vv * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, vv) * w * 2.0
    c = q_vec * np.dot(q_vec, vv) * 2.0
    return a - b + c


def _euler_xyz_from_quat_xyzw(quat_angle: np.ndarray) -> Tuple[float, float, float]:
    quat_angle = np.asarray(quat_angle, dtype=np.float64).reshape(4)
    quat_angle = quat_angle / (np.linalg.norm(quat_angle) + 1e-12)
    x, y, z, w = quat_angle[0], quat_angle[1], quat_angle[2], quat_angle[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = float(np.arctan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = float(np.clip(t2, -1.0, 1.0))
    pitch_y = float(np.arcsin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = float(np.arctan2(t3, t4))
    return roll_x, pitch_y, yaw_z


def _rpy_from_body_xmat(xmat_flat: np.ndarray) -> np.ndarray:
    """由 MuJoCo body xmat (9,) 估计 roll/pitch/yaw (rad)，与常见 ZYX 顺序一致。"""
    r = np.asarray(xmat_flat, dtype=np.float64).reshape(3, 3)
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    pitch = float(np.arctan2(-r[2, 0], np.sqrt(max(r[2, 1] ** 2 + r[2, 2] ** 2, 0.0))))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


@dataclass
class JitterSlipDiagnosticsConfig:
    physics_dt: float
    output_dir: str
    jitter_cutoff_hz: float = 25.0
    foot_body_names: Tuple[str, str] = ("left_leg_ankle_roll_link", "right_leg_ankle_roll_link")
    ground_name_substrings: Tuple[str, ...] = ("floor", "ground", "terrain")
    foot_name_substrings: Tuple[str, ...] = ("foot", "toe", "ankle", "sole")
    ankle_policy_indices: Tuple[int, int, int, int] = (4, 5, 10, 11)
    run_metadata: Optional[Dict[str, Any]] = None


class JitterSlipDiagnostics:
    """每个 mj_step 之后调用 record_after_step；仿真结束后调用 finalize。"""

    def __init__(self, model: mujoco.MjModel, cfg: JitterSlipDiagnosticsConfig):
        self.model = model
        self.cfg = cfg
        self.fs = 1.0 / max(cfg.physics_dt, 1e-12)
        os.makedirs(cfg.output_dir, exist_ok=True)

        self._resolved = self._resolve_geoms_and_bodies()
        self.missing_signals: List[str] = []

        self._foot_body_ids = self._resolved["foot_body_ids"]
        self._foot_geom_to_idx: Dict[int, int] = self._resolved["foot_geom_to_idx"]
        self._ground_geom_ids: List[int] = self._resolved["ground_geom_ids"]
        self._foot_labels = self._resolved["foot_labels"]

        self._print_resolution()

        self._t0_wall = time.time()
        self._base_xy0: Optional[np.ndarray] = None
        self._prev_contact = np.zeros((len(self._foot_body_ids),), dtype=bool)
        self._toggle_count = np.zeros((len(self._foot_body_ids),), dtype=np.int64)

        # 环形缓冲用 list append（步数 ~1e4–1e5 可接受）
        self._buf: Dict[str, List] = {
            "time": [],
            "base_pos": [],
            "base_vel": [],
            "foot_pos": [],
            "foot_vel": [],
            "foot_in_contact": [],
            "foot_normal_force": [],
            "foot_tangential_force": [],
            "foot_slip_speed": [],
            "n_contacts": [],
            "qvel": [],
            "qacc": [],
            "ctrl": [],
            "actuator_force": [],
            "policy_action": [],
            "pd_target": [],
            # sim2sim / GPTPro 扩展（无 track_ctx 时为 NaN；足姿态仍从 MuJoCo 读）
            "motion_time": [],
            "ref_root_pos": [],
            "ref_root_quat_xyzw": [],
            "ref_vel_local": [],
            "ref_ang_vel_local": [],
            "base_quat_xyzw": [],
            "base_rpy": [],
            "base_lin_vel_local": [],
            "base_ang_vel_local": [],
            "vel_err_xy": [],
            "root_xy_err_world": [],
            "dof_pos_policy": [],
            "ankle_tgt": [],
            "ankle_act": [],
            "ankle_err": [],
            "foot_rpy": [],
            "foot_z": [],
            "tau_raw": [],
            "ref_vel_world_xy": [],
            "base_vel_world_xy": [],
            "motion_phase_valid": [],
            "motion_time_unclamped": [],
            "yaw_drift": [],
            "ankle_roll_target": [],
            "ankle_roll_action": [],
            "hip_roll_target": [],
            "hip_roll_action": [],
        }
        self._nu = int(model.nu)

    # --- resolution ---------------------------------------------------------

    def _resolve_geoms_and_bodies(self) -> dict:
        foot_body_names = list(self.cfg.foot_body_names)
        foot_body_ids = []
        for name in foot_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            foot_body_ids.append(int(bid))

        foot_geom_to_idx: Dict[int, int] = {}
        geom_notes: List[str] = []

        for fi, (fname, bid) in enumerate(zip(foot_body_names, foot_body_ids)):
            if bid < 0:
                geom_notes.append(f"foot body missing: {fname}")
                continue
            gadr = int(self.model.body_geomadr[bid])
            gnum = int(self.model.body_geomnum[bid])
            picked = []
            for k in range(gnum):
                gid = gadr + k
                if int(self.model.geom_contype[gid]) == 0 and int(self.model.geom_conaffinity[gid]) == 0:
                    continue
                picked.append(gid)
            if not picked:
                for k in range(gnum):
                    picked.append(gadr + k)
                geom_notes.append(f"{fname}: no nonzero contype/affinity geoms; using all body geoms")
            for gid in picked:
                foot_geom_to_idx[int(gid)] = fi
            geom_notes.append(f"{fname}: collision geoms {picked}")

        ground_ids: set = set()
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            low = name.lower()
            for sub in self.cfg.ground_name_substrings:
                if sub in low:
                    ground_ids.add(gid)
        for literal in ("floor", "ground"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, literal)
            if gid >= 0:
                ground_ids.add(gid)

        if not ground_ids:
            for gid in range(self.model.ngeom):
                if int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                    ground_ids.add(gid)

        auto_foot_geoms: List[str] = []
        if any(b < 0 for b in foot_body_ids):
            pat = re.compile("|".join(re.escape(s) for s in self.cfg.foot_name_substrings), re.I)
            for gid in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                bid = int(self.model.geom_bodyid[gid])
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if pat.search(name) or pat.search(bname):
                    if int(self.model.geom_contype[gid]) == 0 and int(self.model.geom_conaffinity[gid]) == 0:
                        continue
                    auto_foot_geoms.append(f"gid={gid} name={name!r} body={bname!r}")

        foot_labels = []
        for nm in foot_body_names:
            foot_labels.append(nm if nm else f"foot_{len(foot_labels)}")

        return {
            "foot_body_names": foot_body_names,
            "foot_body_ids": foot_body_ids,
            "foot_geom_to_idx": foot_geom_to_idx,
            "ground_geom_ids": sorted(ground_ids),
            "geom_notes": geom_notes,
            "auto_foot_geoms": auto_foot_geoms,
            "foot_labels": foot_labels,
        }

    def _print_resolution(self) -> None:
        print("[JitterSlipDiagnostics] 足端/地面匹配结果（请核对，避免静默误判）:")
        for line in self._resolved["geom_notes"]:
            print("  ", line)
        print("  ground_geom_ids:", self._ground_geom_ids)
        if self._resolved["auto_foot_geoms"]:
            print("  自动按名字匹配的 foot 相关 geom（仅当 body 配置失败时参考）:")
            for s in self._resolved["auto_foot_geoms"][:20]:
                print("   ", s)
            if len(self._resolved["auto_foot_geoms"]) > 20:
                print("    ... 省略其余条目")
        if not self._foot_geom_to_idx:
            print("  [WARN] 未解析到任何足端碰撞 geom；接触力/滑移诊断将失效，请在配置中提供正确 foot body 名称。")
        if not self._ground_geom_ids:
            print("  [WARN] 未解析到地面 geom；请提供名为 floor/ground 的 geom 或在 XML 中便于匹配的命名。")

    # --- lifecycle ----------------------------------------------------------

    def begin_episode(self, data: mujoco.MjData) -> None:
        self._base_xy0 = np.array(data.qpos[0:2], dtype=np.float64).copy()
        self._prev_contact[:] = False
        self._toggle_count[:] = 0

    def record_after_step(
        self,
        data: mujoco.MjData,
        policy_action: np.ndarray,
        pd_target_policy: np.ndarray,
        track_ctx: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        t = float(data.time)
        base_pos = np.array(data.qpos[0:3], dtype=np.float64, copy=True)
        base_vel = np.array(data.qvel[0:3], dtype=np.float64, copy=True)

        n_foot = len(self._foot_body_ids)
        foot_pos = np.zeros((n_foot, 3), dtype=np.float64)
        foot_vel = np.zeros((n_foot, 3), dtype=np.float64)
        vel6 = np.zeros(6, dtype=np.float64)
        for i, bid in enumerate(self._foot_body_ids):
            if bid < 0:
                continue
            foot_pos[i] = data.xpos[bid]
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            foot_vel[i] = vel6[:3]

        foot_in_contact = np.zeros((n_foot,), dtype=bool)
        foot_fn = np.zeros((n_foot,), dtype=np.float64)
        foot_ft = np.zeros((n_foot,), dtype=np.float64)
        foot_slip = np.zeros((n_foot,), dtype=np.float64)
        dominant_normal = [np.array([0.0, 0.0, 1.0], dtype=np.float64) for _ in range(n_foot)]
        best_fn = np.zeros((n_foot,), dtype=np.float64)

        ground_set = set(self._ground_geom_ids)
        f6 = np.zeros(6, dtype=np.float64)

        for ci in range(data.ncon):
            con = data.contact[ci]
            g1, g2 = int(con.geom1), int(con.geom2)
            foot_idx = None
            if g1 in self._foot_geom_to_idx and g2 in ground_set:
                foot_idx = self._foot_geom_to_idx[g1]
            elif g2 in self._foot_geom_to_idx and g1 in ground_set:
                foot_idx = self._foot_geom_to_idx[g2]
            if foot_idx is None:
                continue

            mujoco.mj_contactForce(self.model, data, ci, f6)
            fn = float(f6[0])
            ft_mag = float(np.linalg.norm(f6[1:3]))
            foot_in_contact[foot_idx] = True
            foot_fn[foot_idx] += max(fn, 0.0)
            foot_ft[foot_idx] += ft_mag

            frame = np.array(con.frame, dtype=np.float64).reshape(3, 3)
            n_world = frame[0, :].copy()
            nn = np.linalg.norm(n_world)
            if nn > 1e-9:
                n_world /= nn
            if abs(fn) >= best_fn[foot_idx]:
                best_fn[foot_idx] = abs(fn)
                dominant_normal[foot_idx] = n_world

        for i in range(n_foot):
            if foot_in_contact[i]:
                n = dominant_normal[i]
                if np.linalg.norm(n) < 1e-6:
                    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                v = foot_vel[i]
                v_tan = v - n * np.dot(v, n)
                foot_slip[i] = float(np.linalg.norm(v_tan))
            else:
                foot_slip[i] = 0.0

        for i in range(n_foot):
            if foot_in_contact[i] != bool(self._prev_contact[i]):
                self._toggle_count[i] += 1
            self._prev_contact[i] = foot_in_contact[i]

        qvel = np.array(data.qvel, dtype=np.float64, copy=True)
        qacc = np.array(data.qacc, dtype=np.float64, copy=True)
        ctrl = np.array(data.ctrl, dtype=np.float64, copy=True)

        af = np.array(data.actuator_force, dtype=np.float64, copy=True) if len(data.actuator_force) == self.model.nu else None
        if af is None:
            if "actuator_force" not in self.missing_signals:
                self.missing_signals.append("actuator_force_length_mismatch")

        self._buf["time"].append(t)
        self._buf["base_pos"].append(base_pos)
        self._buf["base_vel"].append(base_vel)
        self._buf["foot_pos"].append(foot_pos)
        self._buf["foot_vel"].append(foot_vel)
        self._buf["foot_in_contact"].append(foot_in_contact.astype(np.float64))
        self._buf["foot_normal_force"].append(foot_fn)
        self._buf["foot_tangential_force"].append(foot_ft)
        self._buf["foot_slip_speed"].append(foot_slip)
        self._buf["n_contacts"].append(float(data.ncon))
        self._buf["qvel"].append(qvel)
        self._buf["qacc"].append(qacc)
        self._buf["ctrl"].append(ctrl)
        self._buf["actuator_force"].append(af if af is not None else np.full_like(ctrl, np.nan))
        self._buf["policy_action"].append(np.asarray(policy_action, dtype=np.float64).reshape(-1))
        pd_tar_vec = np.asarray(pd_target_policy, dtype=np.float64).reshape(-1)
        self._buf["pd_target"].append(pd_tar_vec)
        n_pol = int(pd_tar_vec.shape[0])

        foot_rpy = np.full((2, 3), np.nan, dtype=np.float64)
        foot_z = np.full((2,), np.nan, dtype=np.float64)
        for fi, bid in enumerate(self._foot_body_ids):
            if bid >= 0:
                foot_rpy[fi] = _rpy_from_body_xmat(data.xmat[bid])
                foot_z[fi] = float(data.xpos[bid, 2])

        def _fillnan_track() -> None:
            self._buf["motion_time"].append(np.nan)
            self._buf["ref_root_pos"].append(np.full(3, np.nan, dtype=np.float64))
            self._buf["ref_root_quat_xyzw"].append(np.full(4, np.nan, dtype=np.float64))
            self._buf["ref_vel_local"].append(np.full(3, np.nan, dtype=np.float64))
            self._buf["ref_ang_vel_local"].append(np.full(3, np.nan, dtype=np.float64))
            self._buf["base_quat_xyzw"].append(np.full(4, np.nan, dtype=np.float64))
            self._buf["base_rpy"].append(np.full(3, np.nan, dtype=np.float64))
            self._buf["base_lin_vel_local"].append(np.full(3, np.nan, dtype=np.float64))
            self._buf["base_ang_vel_local"].append(np.full(3, np.nan, dtype=np.float64))
            self._buf["vel_err_xy"].append(np.full(2, np.nan, dtype=np.float64))
            self._buf["root_xy_err_world"].append(np.full(2, np.nan, dtype=np.float64))
            self._buf["dof_pos_policy"].append(np.full(n_pol, np.nan, dtype=np.float64))
            self._buf["ankle_tgt"].append(np.full(4, np.nan, dtype=np.float64))
            self._buf["ankle_act"].append(np.full(4, np.nan, dtype=np.float64))
            self._buf["ankle_err"].append(np.full(4, np.nan, dtype=np.float64))
            self._buf["foot_rpy"].append(foot_rpy)
            self._buf["foot_z"].append(foot_z)
            self._buf["tau_raw"].append(np.full(self._nu, np.nan, dtype=np.float64))
            self._buf["ref_vel_world_xy"].append(np.full(2, np.nan, dtype=np.float64))
            self._buf["base_vel_world_xy"].append(np.full(2, np.nan, dtype=np.float64))
            self._buf["motion_phase_valid"].append(np.nan)
            self._buf["motion_time_unclamped"].append(np.nan)
            self._buf["yaw_drift"].append(np.nan)
            self._buf["ankle_roll_target"].append(np.full(2, np.nan, dtype=np.float64))
            self._buf["ankle_roll_action"].append(np.full(2, np.nan, dtype=np.float64))
            self._buf["hip_roll_target"].append(np.full(2, np.nan, dtype=np.float64))
            self._buf["hip_roll_action"].append(np.full(2, np.nan, dtype=np.float64))

        if track_ctx is None:
            _fillnan_track()
            return

        def _g(key: str, shape: int | Tuple[int, ...]) -> np.ndarray:
            if key not in track_ctx:
                if isinstance(shape, int):
                    return np.full(shape, np.nan, dtype=np.float64)
                return np.full(tuple(shape), np.nan, dtype=np.float64)
            arr = np.asarray(track_ctx[key], dtype=np.float64).reshape(-1)
            if isinstance(shape, int):
                out = np.full(shape, np.nan, dtype=np.float64)
                n = min(shape, arr.size)
                out[:n] = arr[:n]
                return out
            sh = tuple(shape)
            size = int(np.prod(sh))
            out = np.full(sh, np.nan, dtype=np.float64).reshape(-1)
            n = min(size, arr.size)
            out[:n] = arr[:n]
            return out.reshape(sh)

        # 覆盖 base 相关：允许 track_ctx 与当前 MuJoCo 足姿态一起使用
        self._buf["motion_time"].append(float(track_ctx.get("motion_time", np.nan)))
        self._buf["ref_root_pos"].append(_g("ref_root_pos", 3))
        self._buf["ref_root_quat_xyzw"].append(_g("ref_root_quat_xyzw", 4))
        self._buf["ref_vel_local"].append(_g("ref_vel_local", 3))
        self._buf["ref_ang_vel_local"].append(_g("ref_ang_vel_local", 3))
        self._buf["base_quat_xyzw"].append(_g("base_quat_xyzw", 4))
        self._buf["base_rpy"].append(_g("base_rpy", 3))
        self._buf["base_lin_vel_local"].append(_g("base_lin_vel_local", 3))
        self._buf["base_ang_vel_local"].append(_g("base_ang_vel_local", 3))
        self._buf["vel_err_xy"].append(_g("vel_err_xy", 2))
        self._buf["root_xy_err_world"].append(_g("root_xy_err_world", 2))
        self._buf["dof_pos_policy"].append(_g("dof_pos_policy", n_pol))
        self._buf["ankle_tgt"].append(_g("ankle_tgt", 4))
        self._buf["ankle_act"].append(_g("ankle_act", 4))
        self._buf["ankle_err"].append(_g("ankle_err", 4))
        self._buf["foot_rpy"].append(foot_rpy)
        self._buf["foot_z"].append(foot_z)
        self._buf["tau_raw"].append(_g("tau_raw", self._nu))
        self._buf["ref_vel_world_xy"].append(_g("ref_vel_world_xy", 2))
        self._buf["base_vel_world_xy"].append(_g("base_vel_world_xy", 2))
        mv = track_ctx.get("motion_phase_valid", np.nan)
        try:
            self._buf["motion_phase_valid"].append(float(mv))
        except (TypeError, ValueError):
            self._buf["motion_phase_valid"].append(np.nan)
        mtu = track_ctx.get("motion_time_unclamped", np.nan)
        try:
            self._buf["motion_time_unclamped"].append(float(mtu))
        except (TypeError, ValueError):
            self._buf["motion_time_unclamped"].append(np.nan)
        yd = track_ctx.get("yaw_drift", np.nan)
        try:
            self._buf["yaw_drift"].append(float(yd))
        except (TypeError, ValueError):
            self._buf["yaw_drift"].append(np.nan)
        self._buf["ankle_roll_target"].append(_g("ankle_roll_target", 2))
        self._buf["ankle_roll_action"].append(_g("ankle_roll_action", 2))
        self._buf["hip_roll_target"].append(_g("hip_roll_target", 2))
        self._buf["hip_roll_action"].append(_g("hip_roll_action", 2))

    @staticmethod
    def _masked_mean_nan(x: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if mask is not None:
            m = np.asarray(mask, dtype=bool).reshape(-1)
            n = min(x.size, m.size)
            x = x[:n][m[:n]]
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        return float(np.mean(x))

    def _dual_phase_motion_report(
        self,
        physics_dt: float,
        mask_valid_motion: np.ndarray,
        mask_full: np.ndarray,
        ref_vw_xy: np.ndarray,
        base_vw_xy: np.ndarray,
        ref_v_local: np.ndarray,
        base_v_local: np.ndarray,
        root_xy_err_w: np.ndarray,
        yaw_drift: np.ndarray,
        foot_rpy: np.ndarray,
        ank_tgt: np.ndarray,
        ank_act: np.ndarray,
        hip_tgt: np.ndarray,
        hip_act: np.ndarray,
        foot_ic: np.ndarray,
        foot_slip: np.ndarray,
    ) -> Dict[str, Any]:
        """参考 motion 未夹持末帧的区间 vs 整场仿真窗口的统计拆分。"""

        def pack(tag: str, m: np.ndarray) -> Dict[str, Any]:
            if not np.any(m):
                return {"segment": tag, "num_steps": 0, "duration_s": 0.0}
            durs = float(np.sum(m) * physics_dt)
            out: Dict[str, Any] = {
                "segment": tag,
                "num_steps": int(np.sum(m)),
                "duration_s": durs,
                "mean_base_vx_world": self._masked_mean_nan(base_vw_xy[:, 0], m),
                "mean_base_vy_world": self._masked_mean_nan(base_vw_xy[:, 1], m),
                "mean_ref_vx_world": self._masked_mean_nan(ref_vw_xy[:, 0], m),
                "mean_ref_vy_world": self._masked_mean_nan(ref_vw_xy[:, 1], m),
                "mean_base_vx_local": self._masked_mean_nan(base_v_local[:, 0], m),
                "mean_base_vy_local": self._masked_mean_nan(base_v_local[:, 1], m),
                "mean_ref_vx_local": self._masked_mean_nan(ref_v_local[:, 0], m),
                "mean_ref_vy_local": self._masked_mean_nan(ref_v_local[:, 1], m),
                "x_drift": self._masked_mean_nan(root_xy_err_w[:, 0], m),
                "y_drift": self._masked_mean_nan(root_xy_err_w[:, 1], m),
                "yaw_drift": self._masked_mean_nan(yaw_drift, m),
            }
            fr = foot_rpy.reshape(foot_rpy.shape[0], 2, 3)
            out["mean_abs_left_foot_roll"] = self._masked_mean_nan(np.abs(fr[:, 0, 0]), m)
            out["mean_abs_right_foot_roll"] = self._masked_mean_nan(np.abs(fr[:, 1, 0]), m)
            out["mean_left_foot_pitch"] = self._masked_mean_nan(fr[:, 0, 1], m)
            out["mean_right_foot_pitch"] = self._masked_mean_nan(fr[:, 1, 1], m)
            if ank_tgt.shape[1] >= 2:
                out["mean_left_ankle_roll_target"] = self._masked_mean_nan(ank_tgt[:, 0], m)
                out["mean_right_ankle_roll_target"] = self._masked_mean_nan(ank_tgt[:, 1], m)
            if ank_act.shape[1] >= 2:
                out["mean_left_ankle_roll_action"] = self._masked_mean_nan(ank_act[:, 0], m)
                out["mean_right_ankle_roll_action"] = self._masked_mean_nan(ank_act[:, 1], m)
            if hip_tgt.shape[1] >= 2:
                out["mean_left_hip_roll_target"] = self._masked_mean_nan(hip_tgt[:, 0], m)
                out["mean_right_hip_roll_target"] = self._masked_mean_nan(hip_tgt[:, 1], m)
            if hip_act.shape[1] >= 2:
                out["mean_left_hip_roll_action"] = self._masked_mean_nan(hip_act[:, 0], m)
                out["mean_right_hip_roll_action"] = self._masked_mean_nan(hip_act[:, 1], m)
            icm = foot_ic[:, 0] > 0.5
            icm2 = foot_ic[:, 1] > 0.5
            slip_sum = foot_slip[:, 0] * icm.astype(np.float64) + foot_slip[:, 1] * icm2.astype(np.float64)
            ncontact = (icm.astype(np.int32) + icm2.astype(np.int32)).clip(min=1)
            slip_when_contact = slip_sum / ncontact.astype(np.float64)
            out["mean_contact_foot_slip_speed"] = self._masked_mean_nan(slip_when_contact, m)
            out["mean_left_contact"] = self._masked_mean_nan(foot_ic[:, 0], m)
            out["mean_right_contact"] = self._masked_mean_nan(foot_ic[:, 1], m)
            return out

        rep_valid = pack("valid_reference_motion_not_clamped", mask_valid_motion)
        rep_full = pack("full_simulation_window", mask_full)
        return {"valid_reference_segment": rep_valid, "full_run": rep_full}

    # --- finalize -----------------------------------------------------------

    def finalize(self) -> str:
        out_dir = self.cfg.output_dir
        if not self._buf["time"]:
            summary = {
                "diagnosis": "insufficient_data",
                "reasons": ["no physics steps recorded"],
                "missing_signals": self.missing_signals,
                "output_dir": out_dir,
            }
            with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(_safe_json(summary), f, indent=2, ensure_ascii=False)
            return out_dir

        time_s = np.asarray(self._buf["time"], dtype=np.float64)
        base_pos = np.stack(self._buf["base_pos"], axis=0)
        base_vel = np.stack(self._buf["base_vel"], axis=0)
        foot_pos = np.stack(self._buf["foot_pos"], axis=0)
        foot_vel = np.stack(self._buf["foot_vel"], axis=0)
        foot_ic = np.stack(self._buf["foot_in_contact"], axis=0)
        foot_fn = np.stack(self._buf["foot_normal_force"], axis=0)
        foot_ft = np.stack(self._buf["foot_tangential_force"], axis=0)
        foot_slip = np.stack(self._buf["foot_slip_speed"], axis=0)
        n_con = np.asarray(self._buf["n_contacts"], dtype=np.float64)
        qvel = np.stack(self._buf["qvel"], axis=0)
        qacc = np.stack(self._buf["qacc"], axis=0)
        ctrl = np.stack(self._buf["ctrl"], axis=0)
        af = np.stack(self._buf["actuator_force"], axis=0)
        act = np.stack(self._buf["policy_action"], axis=0)
        pd_tar = np.stack(self._buf["pd_target"], axis=0)
        motion_time_arr = np.asarray(self._buf["motion_time"], dtype=np.float64)
        ref_root_pos = np.stack(self._buf["ref_root_pos"], axis=0)
        ref_root_quat_xyzw = np.stack(self._buf["ref_root_quat_xyzw"], axis=0)
        ref_vel_local = np.stack(self._buf["ref_vel_local"], axis=0)
        ref_ang_vel_local = np.stack(self._buf["ref_ang_vel_local"], axis=0)
        base_quat_xyzw = np.stack(self._buf["base_quat_xyzw"], axis=0)
        base_rpy = np.stack(self._buf["base_rpy"], axis=0)
        base_lin_vel_local = np.stack(self._buf["base_lin_vel_local"], axis=0)
        base_ang_vel_local = np.stack(self._buf["base_ang_vel_local"], axis=0)
        vel_err_xy = np.stack(self._buf["vel_err_xy"], axis=0)
        root_xy_err_world = np.stack(self._buf["root_xy_err_world"], axis=0)
        dof_pos_policy = np.stack(self._buf["dof_pos_policy"], axis=0)
        ankle_tgt = np.stack(self._buf["ankle_tgt"], axis=0)
        ankle_act = np.stack(self._buf["ankle_act"], axis=0)
        ankle_err = np.stack(self._buf["ankle_err"], axis=0)
        foot_rpy_t = np.stack(self._buf["foot_rpy"], axis=0)
        foot_z_t = np.stack(self._buf["foot_z"], axis=0)
        tau_raw = np.stack(self._buf["tau_raw"], axis=0)
        ref_vel_world_xy_t = np.stack(self._buf["ref_vel_world_xy"], axis=0)
        base_vel_world_xy_t = np.stack(self._buf["base_vel_world_xy"], axis=0)
        motion_phase_valid_t = np.asarray(self._buf["motion_phase_valid"], dtype=np.float64)
        yaw_drift_t = np.asarray(self._buf["yaw_drift"], dtype=np.float64)
        ankle_roll_target_t = np.stack(self._buf["ankle_roll_target"], axis=0)
        ankle_roll_action_t = np.stack(self._buf["ankle_roll_action"], axis=0)
        hip_roll_target_t = np.stack(self._buf["hip_roll_target"], axis=0)
        hip_roll_action_t = np.stack(self._buf["hip_roll_action"], axis=0)
        if np.any(np.isnan(af)):
            if "actuator_force_contains_nan" not in self.missing_signals:
                self.missing_signals.append("actuator_force_contains_nan")

        dt = float(self.cfg.physics_dt)
        duration = float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0

        # 基座水平漂移（净位移）
        if self._base_xy0 is not None:
            drift_xy = np.linalg.norm(base_pos[-1, :2] - self._base_xy0)
        else:
            drift_xy = float(np.linalg.norm(base_pos[-1, :2] - base_pos[0, :2]))

        per_foot = []
        for i, label in enumerate(self._foot_labels):
            ic = foot_ic[:, i] > 0.5
            slip_when = np.where(ic, foot_slip[:, i], 0.0)
            int_slip = float(np.sum(slip_when * dt))
            mean_slip = float(np.mean(slip_when[ic])) if np.any(ic) else 0.0
            max_slip = float(np.max(foot_slip[:, i])) if foot_slip.size else 0.0

            fn_s = foot_fn[:, i]
            ft_s = foot_ft[:, i]
            slip_s = foot_slip[:, i]

            fn_hf, m1 = _highpass_series(fn_s, self.fs, self.cfg.jitter_cutoff_hz)
            ft_hf, m2 = _highpass_series(ft_s, self.fs, self.cfg.jitter_cutoff_hz)
            sp_hf, m3 = _highpass_series(slip_s, self.fs, self.cfg.jitter_cutoff_hz)

            per_foot.append(
                {
                    "label": label,
                    "contact_toggle_count": int(self._toggle_count[i]),
                    "contact_toggle_rate_hz": float(self._toggle_count[i]) / max(duration, 1e-6),
                    "mean_slip_speed_m_s": mean_slip,
                    "max_slip_speed_m_s": max_slip,
                    "integrated_slip_distance_m": int_slip,
                    "normal_force_rms_n": _rms(fn_s),
                    "normal_force_hf_rms_n": _rms(fn_hf),
                    "normal_force_hf_method": m1,
                    "tangential_force_rms_n": _rms(ft_s),
                    "tangential_force_hf_rms_n": _rms(ft_hf),
                    "tangential_force_hf_method": m2,
                    "foot_slip_speed_rms_m_s": _rms(slip_s),
                    "foot_slip_speed_hf_rms_m_s": _rms(sp_hf),
                    "foot_slip_speed_hf_method": m3,
                }
            )

        act_norm = np.linalg.norm(act, axis=1)
        ctrl_norm = np.linalg.norm(ctrl, axis=1)
        act_hf, ma = _highpass_series(act_norm, self.fs, self.cfg.jitter_cutoff_hz)
        ctrl_hf, mc = _highpass_series(ctrl_norm, self.fs, self.cfg.jitter_cutoff_hz)

        summary_core = {
            "duration_s": duration,
            "physics_dt_s": dt,
            "sample_rate_hz": self.fs,
            "jitter_cutoff_hz": self.cfg.jitter_cutoff_hz,
            "base_drift_xy_m": float(drift_xy),
            "num_steps": int(time_s.size),
            "foot_resolution": self._resolved,
            "per_foot": per_foot,
            "policy_action_norm_rms": _rms(act_norm),
            "policy_action_norm_hf_rms": _rms(act_hf),
            "policy_action_hf_method": ma,
            "ctrl_norm_rms": _rms(ctrl_norm),
            "ctrl_norm_hf_rms": _rms(ctrl_hf),
            "ctrl_hf_method": mc,
            "contact_count_mean": float(np.mean(n_con)) if n_con.size else 0.0,
            "wall_clock_s": float(time.time() - self._t0_wall),
        }

        diagnosis, reasons = self._diagnose(summary_core, per_foot, drift_xy, duration, act_norm, ctrl_norm)

        sim2sim_summary = self._compute_sim2sim_summary(
            dt=dt,
            duration=duration,
            motion_time_arr=motion_time_arr,
            ref_vel_local=ref_vel_local,
            base_lin_vel_local=base_lin_vel_local,
            vel_err_xy=vel_err_xy,
            root_xy_err_world=root_xy_err_world,
            foot_rpy_t=foot_rpy_t,
            ankle_err=ankle_err,
            tau_raw=tau_raw,
            ctrl=ctrl,
        )

        mask_valid_motion = np.isfinite(motion_phase_valid_t) & (motion_phase_valid_t > 0.5)
        mask_full = (
            np.isfinite(motion_time_arr)
            & np.isfinite(vel_err_xy[:, 0])
            & np.isfinite(vel_err_xy[:, 1])
        )
        sim2sim_motion_phase = self._dual_phase_motion_report(
            dt,
            mask_valid_motion,
            mask_full,
            ref_vel_world_xy_t,
            base_vel_world_xy_t,
            ref_vel_local,
            base_lin_vel_local,
            root_xy_err_world,
            yaw_drift_t,
            foot_rpy_t,
            ankle_roll_target_t,
            ankle_roll_action_t,
            hip_roll_target_t,
            hip_roll_action_t,
            foot_ic,
            foot_slip,
        )

        summary = {
            **summary_core,
            "diagnosis": diagnosis,
            "reasons": reasons,
            "missing_signals": list(dict.fromkeys(self.missing_signals)),
            "sim2sim_summary": sim2sim_summary,
            "sim2sim_motion_phase": sim2sim_motion_phase,
            "run_metadata": self.cfg.run_metadata,
        }
        vr = sim2sim_motion_phase.get("valid_reference_segment") or {}
        summary["motion_phase_quick_view"] = {
            "valid_motion_duration_s": vr.get("duration_s"),
            "valid_mean_base_vx_local": vr.get("mean_base_vx_local"),
            "valid_mean_base_vx_world": vr.get("mean_base_vx_world"),
            "valid_mean_ref_vx_world": vr.get("mean_ref_vx_world"),
            "valid_x_drift_mean_m": vr.get("x_drift"),
            "valid_yaw_drift_mean_rad": vr.get("yaw_drift"),
            "valid_mean_abs_left_foot_roll": vr.get("mean_abs_left_foot_roll"),
            "valid_mean_abs_right_foot_roll": vr.get("mean_abs_right_foot_roll"),
            "valid_mean_left_ankle_roll_target": vr.get("mean_left_ankle_roll_target"),
            "valid_mean_right_ankle_roll_target": vr.get("mean_right_ankle_roll_target"),
            "valid_mean_left_ankle_roll_action": vr.get("mean_left_ankle_roll_action"),
            "valid_mean_right_ankle_roll_action": vr.get("mean_right_ankle_roll_action"),
            "full_window_duration_s": (sim2sim_motion_phase.get("full_run") or {}).get("duration_s"),
        }

        np.savez_compressed(
            os.path.join(out_dir, "raw_timeseries.npz"),
            time=time_s,
            base_pos=base_pos,
            base_vel=base_vel,
            foot_pos=foot_pos,
            foot_vel=foot_vel,
            foot_in_contact=foot_ic,
            foot_normal_force=foot_fn,
            foot_tangential_force=foot_ft,
            foot_slip_speed=foot_slip,
            n_contacts=n_con,
            qvel=qvel,
            qacc=qacc,
            ctrl=ctrl,
            actuator_force=af,
            policy_action=act,
            pd_target=pd_tar,
            foot_labels=np.array(self._foot_labels, dtype=object),
            motion_time=motion_time_arr,
            ref_root_pos=ref_root_pos,
            ref_root_quat_xyzw=ref_root_quat_xyzw,
            ref_vel_local=ref_vel_local,
            ref_ang_vel_local=ref_ang_vel_local,
            base_quat_xyzw=base_quat_xyzw,
            base_rpy=base_rpy,
            base_lin_vel_local=base_lin_vel_local,
            base_ang_vel_local=base_ang_vel_local,
            vel_err_xy=vel_err_xy,
            root_xy_err_world=root_xy_err_world,
            dof_pos_policy=dof_pos_policy,
            ankle_tgt=ankle_tgt,
            ankle_act=ankle_act,
            ankle_err=ankle_err,
            foot_rpy=foot_rpy_t,
            foot_z=foot_z_t,
            tau_raw=tau_raw,
            ref_vel_world_xy=ref_vel_world_xy_t,
            base_vel_world_xy=base_vel_world_xy_t,
            motion_phase_valid=motion_phase_valid_t,
            yaw_drift=yaw_drift_t,
            ankle_roll_target=ankle_roll_target_t,
            ankle_roll_action=ankle_roll_action_t,
            hip_roll_target=hip_roll_target_t,
            hip_roll_action=hip_roll_action_t,
        )

        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(_safe_json(summary), f, indent=2, ensure_ascii=False)

        self._save_plots(
            time_s=time_s,
            base_pos=base_pos,
            foot_slip=foot_slip,
            foot_fn=foot_fn,
            foot_ft=foot_ft,
            foot_ic=foot_ic,
            act=act,
            ctrl=ctrl,
            out_dir=out_dir,
            ref_vel_local=ref_vel_local,
            base_lin_vel_local=base_lin_vel_local,
            vel_err_xy=vel_err_xy,
            base_rpy=base_rpy,
            foot_rpy_t=foot_rpy_t,
            ankle_tgt=ankle_tgt,
            ankle_act=ankle_act,
            sim2sim_summary=sim2sim_summary,
        )
        self._save_gptpro_report(out_dir, sim2sim_summary, summary_core, diagnosis, reasons)

        print(f"[JitterSlipDiagnostics] 已写入: {out_dir}")
        print(f"[JitterSlipDiagnostics] diagnosis={diagnosis}")
        qv = summary.get("motion_phase_quick_view") or {}
        if qv:
            print("[JitterSlipDiagnostics] motion_phase_quick_view (valid ref segment vs full run 见 summary.json):")
            for kk, vv in qv.items():
                print(f"  {kk}: {vv}")
        for r in reasons:
            print("  -", r)
        return out_dir

    def _compute_sim2sim_summary(
        self,
        dt: float,
        duration: float,
        motion_time_arr: np.ndarray,
        ref_vel_local: np.ndarray,
        base_lin_vel_local: np.ndarray,
        vel_err_xy: np.ndarray,
        root_xy_err_world: np.ndarray,
        foot_rpy_t: np.ndarray,
        ankle_err: np.ndarray,
        tau_raw: np.ndarray,
        ctrl: np.ndarray,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "has_track_ctx": False,
            "note": "vel_err_xy matches cp_fdd_V1 training: local_ref_xy - base_lin_vel_xy (mixed frames).",
        }
        valid_track = np.isfinite(motion_time_arr) & np.isfinite(vel_err_xy[:, 0]) & np.isfinite(vel_err_xy[:, 1])
        if not np.any(valid_track):
            out["note"] += " No valid track_ctx rows; pass track_ctx from sim script for velocity/ankle diagnostics."
            if foot_rpy_t.size and np.any(np.isfinite(foot_rpy_t)):
                fr = foot_rpy_t.reshape(-1, 2, 3)
                mask = np.isfinite(fr[:, 0, 0])
                if np.any(mask):
                    out["foot_roll_mean_left_rad"] = float(np.nanmean(fr[mask, 0, 0]))
                    out["foot_roll_mean_right_rad"] = float(np.nanmean(fr[mask, 1, 0]))
                    out["foot_roll_std_left_rad"] = float(np.nanstd(fr[mask, 0, 0]))
                    out["foot_roll_std_right_rad"] = float(np.nanstd(fr[mask, 1, 0]))
            return out

        out["has_track_ctx"] = True
        ve = vel_err_xy[valid_track]
        out["vel_xy_rmse_m_s"] = float(np.sqrt(np.mean(np.sum(ve * ve, axis=1))))
        out["vel_xy_err_mean"] = [float(np.mean(ve[:, 0])), float(np.mean(ve[:, 1]))]

        ref_v = ref_vel_local[valid_track]
        base_v = base_lin_vel_local[valid_track]
        ref_speed_xy = np.linalg.norm(ref_v[:, :2], axis=1)
        stationary = ref_speed_xy < 0.1
        out["fraction_ref_stationary_xy"] = float(np.mean(stationary)) if stationary.size else 0.0
        if np.any(stationary):
            bs = base_v[stationary]
            out["stationary_mean_base_xy_speed_m_s"] = float(np.mean(np.linalg.norm(bs[:, :2], axis=1)))
            out["stationary_mean_base_vx_m_s"] = float(np.mean(bs[:, 0]))
            out["stationary_mean_base_vy_m_s"] = float(np.mean(bs[:, 1]))
        else:
            out["stationary_mean_base_xy_speed_m_s"] = None
            out["stationary_mean_base_vx_m_s"] = None
            out["stationary_mean_base_vy_m_s"] = None

        fr = foot_rpy_t[valid_track]
        if fr.size:
            out["foot_roll_mean_left_rad"] = float(np.nanmean(fr[:, 0, 0]))
            out["foot_roll_mean_right_rad"] = float(np.nanmean(fr[:, 1, 0]))
            out["foot_roll_std_left_rad"] = float(np.nanstd(fr[:, 0, 0]))
            out["foot_roll_std_right_rad"] = float(np.nanstd(fr[:, 1, 0]))
            out["foot_pitch_mean_left_rad"] = float(np.nanmean(fr[:, 0, 1]))
            out["foot_pitch_mean_right_rad"] = float(np.nanmean(fr[:, 1, 1]))

        ae = ankle_err[valid_track]
        if ae.size and np.any(np.isfinite(ae)):
            flat = ae.reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size:
                out["ankle_err_rms_rad"] = float(_rms(flat))
            labels = ("L_pitch", "L_roll", "R_pitch", "R_roll")
            out["ankle_err_mean_rad"] = {
                labels[i]: float(np.nanmean(ae[:, i])) for i in range(min(4, ae.shape[1]))
            }

        rxy = root_xy_err_world[valid_track]
        if rxy.size and np.any(np.isfinite(rxy)):
            finite = np.isfinite(rxy[:, 0]) & np.isfinite(rxy[:, 1])
            if np.any(finite):
                rr = rxy[finite]
                out["root_xy_err_world_rmse_m"] = float(np.sqrt(np.mean(np.sum(rr * rr, axis=1))))

        tor_valid = (
            valid_track
            & np.isfinite(tau_raw).all(axis=1)
            & np.isfinite(ctrl).all(axis=1)
        )
        if np.any(tor_valid):
            d = np.abs(tau_raw[tor_valid] - ctrl[tor_valid])
            out["tau_raw_vs_ctrl_l2_mean"] = float(np.mean(np.linalg.norm(d, axis=1)))
            out["tau_raw_vs_ctrl_l2_max"] = float(np.max(np.linalg.norm(d, axis=1)))

        out["duration_s"] = duration
        out["physics_dt_s"] = dt
        return out

    def _save_gptpro_report(
        self,
        out_dir: str,
        sim2sim_summary: Dict[str, Any],
        summary_core: Dict[str, Any],
        diagnosis: str,
        reasons: List[str],
    ) -> None:
        meta = self.cfg.run_metadata or {}
        lines = [
            "# Sim2Sim 诊断包（供 GPTPro）",
            "",
            "## 运行元数据",
            "",
            "```json",
            json.dumps(_safe_json(meta), indent=2, ensure_ascii=False),
            "```",
            "",
            "## 关键结论（启发式）",
            "",
            f"- **diagnosis**: `{diagnosis}`",
        ]
        for r in reasons:
            lines.append(f"- {r}")
        lines.extend(
            [
                "",
                "## sim2sim 量化摘要",
                "",
                "```json",
                json.dumps(_safe_json(sim2sim_summary), indent=2, ensure_ascii=False),
                "```",
                "",
                "## 建议优先查看",
                "",
                "- `raw_timeseries.npz`: `vel_err_xy`, `base_lin_vel_local`, `ref_vel_local`（原地踏步漂移 / 速度跟踪）",
                "- `foot_rpy`（左右脚 roll/pitch，内翻可看 roll 符号与幅值）",
                "- `ankle_tgt` vs `ankle_act`、`ankle_err`（PD 跟踪与踝目标）",
                "- `tau_raw` vs `ctrl`（力矩限幅是否经常生效）",
                "- `foot_slip_speed`, `foot_normal_force`（滑移与接触）",
                "",
                "## 文件列表",
                "",
                "- `summary.json` — 全文统计 + `sim2sim_summary`",
                "- `raw_timeseries.npz` — 每 physics step 时序",
                "- `plots.png` — 曲线总览",
                "",
            ]
        )
        path_md = os.path.join(out_dir, "gptpro_report.md")
        with open(path_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _diagnose(
        self,
        summary_core: dict,
        per_foot: List[dict],
        drift_xy: float,
        duration: float,
        act_norm: np.ndarray,
        ctrl_norm: np.ndarray,
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []

        if duration < 0.5 or summary_core["num_steps"] < 10:
            reasons.append(f"insufficient duration/steps (T={duration:.3f}s, n={summary_core['num_steps']})")
            return "insufficient_data", reasons

        if not self._foot_geom_to_idx or not self._ground_geom_ids:
            reasons.append("foot geom or ground geom not resolved; contact metrics unreliable")
            return "insufficient_data", reasons

        # 阈值：保守、可解释；非物理标定
        DRIFT_STRONG = 0.08
        DRIFT_MILD = 0.03
        SLIP_INT_STRONG = 0.12
        TOGGLE_RATE_HIGH = 5.0
        HF_RATIO_FORCE = 0.28
        HF_SLIP = 0.04
        ACT_HF_RATIO = 0.30

        if drift_xy >= DRIFT_STRONG:
            reasons.append(f"base drift reached {drift_xy:.3f} m (horizontal net displacement)")
        elif drift_xy >= DRIFT_MILD:
            reasons.append(f"base drift moderate {drift_xy:.3f} m")

        hf_markers = 0
        chatter = False
        slip_accum = False

        for pf in per_foot:
            label = pf["label"]
            if pf["integrated_slip_distance_m"] >= SLIP_INT_STRONG:
                reasons.append(f"{label} contact slip integrated to {pf['integrated_slip_distance_m']:.3f} m")
                slip_accum = True
            tr = pf["contact_toggle_rate_hz"]
            if tr >= TOGGLE_RATE_HIGH:
                reasons.append(f"{label} contact toggled ~{int(round(tr * max(duration, 1e-6)))} times over {duration:.1f}s ({tr:.1f} Hz)")
                chatter = True
                hf_markers += 1
            rms = max(pf["normal_force_rms_n"], 1e-9)
            if pf["normal_force_hf_rms_n"] / rms >= HF_RATIO_FORCE and rms > 1.0:
                reasons.append(f"{label} normal force high-frequency RMS ratio high ({pf['normal_force_hf_rms_n'] / rms:.2f})")
                hf_markers += 1
            rms_t = max(pf["tangential_force_rms_n"], 1e-9)
            if pf["tangential_force_hf_rms_n"] / rms_t >= HF_RATIO_FORCE and rms_t > 0.5:
                reasons.append(f"{label} tangential force high-frequency RMS ratio high ({pf['tangential_force_hf_rms_n'] / rms_t:.2f})")
                hf_markers += 1
            if pf["foot_slip_speed_hf_rms_m_s"] >= HF_SLIP:
                reasons.append(f"{label} slip_speed high-frequency RMS is {pf['foot_slip_speed_hf_rms_m_s']:.4f} m/s")
                hf_markers += 1

        act_hf_series = _highpass_series(act_norm, self.fs, self.cfg.jitter_cutoff_hz)[0]
        act_rms = _rms(act_norm)
        act_hf_rms = _rms(act_hf_series)
        if act_rms > 1e-6 and act_hf_rms / act_rms >= ACT_HF_RATIO:
            reasons.append(f"policy action L2 norm high-frequency RMS ratio high ({act_hf_rms / act_rms:.2f})")
            hf_markers += 1

        ctrl_hf_series = _highpass_series(ctrl_norm, self.fs, self.cfg.jitter_cutoff_hz)[0]
        ctrl_rms = _rms(ctrl_norm)
        ctrl_hf_rms = _rms(ctrl_hf_series)
        if ctrl_rms > 1e-3 and ctrl_hf_rms / ctrl_rms >= ACT_HF_RATIO:
            reasons.append(f"control torque L2 norm high-frequency RMS ratio high ({ctrl_hf_rms / ctrl_rms:.2f})")
            hf_markers += 1

        strong_drift = drift_xy >= DRIFT_STRONG

        if not reasons:
            reasons.append("no strong diagnostic triggers; drift and HF contact/action cues appear low")

        # 保守规则：漂移 +（高频接触/控制振荡 或 接触颤振）+（滑移累积或切向/滑移高频）→ 更倾向 jitter
        if strong_drift and hf_markers >= 2 and (slip_accum or chatter):
            return "likely_high_frequency_jitter_induced_slip", reasons
        if strong_drift and hf_markers >= 3:
            return "likely_high_frequency_jitter_induced_slip", reasons
        if strong_drift and hf_markers >= 1:
            return "possible_high_frequency_jitter", reasons
        if drift_xy >= DRIFT_MILD and hf_markers >= 2:
            return "possible_high_frequency_jitter", reasons
        if strong_drift and hf_markers == 0:
            reasons.append("drift present but weak HF coupling in logged signals (may be other causes)")
            return "possible_high_frequency_jitter", reasons
        # 基座漂移不明显时，仍可能出现足端高频滑移/力振荡（肉眼难辨）
        if slip_accum and hf_markers >= 2:
            if strong_drift or chatter:
                return "likely_high_frequency_jitter_induced_slip", reasons
            return "possible_high_frequency_jitter", reasons
        if slip_accum and hf_markers >= 1:
            return "possible_high_frequency_jitter", reasons
        return "unlikely_high_frequency_jitter", reasons

    def _save_plots(
        self,
        *,
        time_s: np.ndarray,
        base_pos: np.ndarray,
        foot_slip: np.ndarray,
        foot_fn: np.ndarray,
        foot_ft: np.ndarray,
        foot_ic: np.ndarray,
        act: np.ndarray,
        ctrl: np.ndarray,
        out_dir: str,
        ref_vel_local: Optional[np.ndarray] = None,
        base_lin_vel_local: Optional[np.ndarray] = None,
        vel_err_xy: Optional[np.ndarray] = None,
        base_rpy: Optional[np.ndarray] = None,
        foot_rpy_t: Optional[np.ndarray] = None,
        ankle_tgt: Optional[np.ndarray] = None,
        ankle_act: Optional[np.ndarray] = None,
        sim2sim_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            self.missing_signals.append(f"matplotlib_unavailable:{e}")
            return

        has_track = bool(sim2sim_summary and sim2sim_summary.get("has_track_ctx"))
        labels = self._foot_labels
        nrows = 13 if has_track else 6
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 1.95 * nrows), sharex=True)
        row = 0

        axes[row].plot(time_s, base_pos[:, 0], label="base x")
        axes[row].plot(time_s, base_pos[:, 1], label="base y")
        axes[row].set_ylabel("m")
        axes[row].legend(loc="upper right", fontsize=8)
        axes[row].set_title("Base horizontal position")
        row += 1

        if has_track and ref_vel_local is not None and base_lin_vel_local is not None:
            axes[row].plot(time_s, ref_vel_local[:, 0], label="ref vx (local ref)", alpha=0.85)
            axes[row].plot(time_s, ref_vel_local[:, 1], label="ref vy (local ref)", alpha=0.85)
            axes[row].plot(time_s, base_lin_vel_local[:, 0], "--", label="base vx (local base)", alpha=0.85)
            axes[row].plot(time_s, base_lin_vel_local[:, 1], "--", label="base vy (local base)", alpha=0.85)
            axes[row].set_ylabel("m/s")
            axes[row].legend(loc="upper right", fontsize=7)
            axes[row].set_title("Reference vs actual linear velocity (training-style local frames)")
            row += 1

            if vel_err_xy is not None:
                axes[row].plot(time_s, vel_err_xy[:, 0], label="err vx")
                axes[row].plot(time_s, vel_err_xy[:, 1], label="err vy")
                axes[row].set_ylabel("m/s")
                axes[row].legend(loc="upper right", fontsize=8)
                axes[row].set_title("vel_err_xy = ref_vel_local_xy - base_vel_local_xy")
                row += 1

            if base_rpy is not None:
                axes[row].plot(time_s, base_rpy[:, 0], label="roll")
                axes[row].plot(time_s, base_rpy[:, 1], label="pitch")
                axes[row].plot(time_s, base_rpy[:, 2], label="yaw")
                axes[row].set_ylabel("rad")
                axes[row].legend(loc="upper right", fontsize=8)
                axes[row].set_title("Base roll / pitch / yaw")
                row += 1

            if foot_rpy_t is not None and foot_rpy_t.ndim == 3:
                axes[row].plot(time_s, foot_rpy_t[:, 0, 0], label="L foot roll")
                axes[row].plot(time_s, foot_rpy_t[:, 1, 0], label="R foot roll")
                axes[row].set_ylabel("rad")
                axes[row].legend(loc="upper right", fontsize=8)
                axes[row].set_title("Foot roll (body xmat RPY; check sign for supination/pronation)")
                row += 1

                axes[row].plot(time_s, foot_rpy_t[:, 0, 1], label="L foot pitch")
                axes[row].plot(time_s, foot_rpy_t[:, 1, 1], label="R foot pitch")
                axes[row].set_ylabel("rad")
                axes[row].legend(loc="upper right", fontsize=8)
                axes[row].set_title("Foot pitch")
                row += 1

            if ankle_tgt is not None and ankle_act is not None and ankle_tgt.shape[1] >= 4:
                nm = ("L pitch", "L roll", "R pitch", "R roll")
                for j in range(4):
                    axes[row].plot(time_s, ankle_tgt[:, j], label=f"tar {nm[j]}", alpha=0.8)
                    axes[row].plot(time_s, ankle_act[:, j], "--", label=f"act {nm[j]}", alpha=0.75)
                axes[row].set_ylabel("rad")
                axes[row].legend(loc="upper right", fontsize=6, ncol=2)
                axes[row].set_title("Ankle joint target vs actual (policy order)")
                row += 1

        while row < nrows - 5:
            axes[row].set_visible(False)
            row += 1

        for i, lab in enumerate(labels):
            axes[row].plot(time_s, foot_slip[:, i], label=lab)
        axes[row].set_ylabel("m/s")
        axes[row].legend(loc="upper right", fontsize=8)
        axes[row].set_title("Foot slip speed (contact only)")
        row += 1

        for i, lab in enumerate(labels):
            axes[row].plot(time_s, foot_fn[:, i], label=lab)
        axes[row].set_ylabel("N (sum)")
        axes[row].legend(loc="upper right", fontsize=8)
        axes[row].set_title("Foot normal force (summed contacts)")
        row += 1

        for i, lab in enumerate(labels):
            axes[row].plot(time_s, foot_ft[:, i], label=lab)
        axes[row].set_ylabel("N (sum)")
        axes[row].legend(loc="upper right", fontsize=8)
        axes[row].set_title("Foot tangential force magnitude (summed)")
        row += 1

        for i, lab in enumerate(labels):
            axes[row].step(time_s, foot_ic[:, i], where="post", label=lab)
        axes[row].set_ylabel("0/1")
        axes[row].set_ylim(-0.1, 1.1)
        axes[row].legend(loc="upper right", fontsize=8)
        axes[row].set_title("Foot contact timeline")
        row += 1

        act_norm_t = np.linalg.norm(act, axis=1)
        ctrl_norm_t = np.linalg.norm(ctrl, axis=1)
        axes[row].plot(time_s, act_norm_t, color="C0", label="||policy action||")
        ax_r = axes[row].twinx()
        ax_r.plot(time_s, ctrl_norm_t, color="C1", alpha=0.85, label="||ctrl torque||")
        axes[row].set_ylabel("||action||", color="C0")
        ax_r.set_ylabel("||ctrl||", color="C1")
        axes[row].tick_params(axis="y", labelcolor="C0")
        ax_r.tick_params(axis="y", labelcolor="C1")
        axes[row].set_xlabel("sim time (s)")
        h1, l1 = axes[row].get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax_r.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        axes[row].set_title("Policy action norm & control torque norm")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plots.png"), dpi=120)
        plt.close(fig)
