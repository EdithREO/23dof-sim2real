#!/usr/bin/env python3
"""Render the existing G1 ONNX/CSV sim2sim runner to an MP4."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = BUNDLE_ROOT / "tw_g1_mujoco_rmg" / "scripts" / "g1_mujoco_sim_rmg_onnx_csv.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("g1_sim2sim_video_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=4.8)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-16.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    return args, runner_args


def main() -> None:
    args, runner_args = _parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("fps, width and height must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.video.parent.mkdir(parents=True, exist_ok=True)

    runner = _load_runner()
    sim_holder: dict[str, object] = {}
    render_info: dict[str, object] = {"frames": 0}
    original_init = runner.cp.MujocoRMGSim.__init__
    original_loop = runner.cp.MujocoRMGSim._run_sim_loop

    def capture_init(sim, *init_args, **init_kwargs) -> None:
        original_init(sim, *init_args, **init_kwargs)
        sim_holder["sim"] = sim

    def render_loop(sim, headless: bool, viewer=None) -> None:
        del headless, viewer
        sim.model.vis.global_.offwidth = max(
            int(sim.model.vis.global_.offwidth), args.width
        )
        sim.model.vis.global_.offheight = max(
            int(sim.model.vis.global_.offheight), args.height
        )
        renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = float(args.distance)
        camera.azimuth = float(args.azimuth)
        camera.elevation = float(args.elevation)
        next_frame_t = 0.0
        frame_count = 0
        writer = imageio.get_writer(
            str(args.video),
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )

        def append_frame() -> None:
            nonlocal frame_count
            base = np.asarray(sim.data.qpos[:3], dtype=np.float64)
            camera.lookat[:] = (base[0], base[1], max(0.55, 0.52 * base[2]))
            renderer.update_scene(sim.data, camera=camera)
            writer.append_data(renderer.render())
            frame_count += 1

        try:
            append_frame()
            next_frame_t += 1.0 / float(args.fps)
            while True:
                u_t = sim.motion_time_offset + float(sim.episode_length_buf.item()) * sim.dt
                sim._last_motion_time = float(u_t)
                if not sim._should_continue_run(u_t, True):
                    break
                motion_time = min(u_t, sim.motion_len - 0.01)
                sim._step_policy(
                    torch.tensor([motion_time], device=sim.device, dtype=torch.float32),
                    u_t,
                )
                sim.episode_length_buf += 1
                elapsed = float(sim.episode_length_buf.item()) * sim.dt
                if elapsed + 1.0e-9 >= next_frame_t:
                    append_frame()
                    next_frame_t += 1.0 / float(args.fps)
                if sim._should_stop_run_after_fall(u_t):
                    fallen, reason = sim._fall_termination_reason()
                    sim._run_stop_reason = reason if fallen else "fall"
                    print(
                        f"[sim2sim-video] fall at {u_t:.2f}s "
                        f"reason={sim._run_stop_reason}",
                        flush=True,
                    )
                    break
                u_t_after = sim.motion_time_offset + float(sim.episode_length_buf.item()) * sim.dt
                sim._last_motion_time = float(u_t_after)
                if not sim._should_continue_run(u_t_after, True):
                    sim._print_run_stop_reason(u_t_after)
                    break
        finally:
            writer.close()
            renderer.close()
            render_info.update(
                frames=frame_count,
                duration_s=frame_count / float(args.fps),
                stop_reason=str(getattr(sim, "_run_stop_reason", "")),
                last_motion_time_s=float(getattr(sim, "_last_motion_time", 0.0)),
            )

    runner.cp.MujocoRMGSim.__init__ = capture_init
    runner.cp.MujocoRMGSim._run_sim_loop = render_loop
    try:
        sys.argv = [str(RUNNER), *runner_args, "--headless"]
        runner.main()
    finally:
        runner.cp.MujocoRMGSim.__init__ = original_init
        runner.cp.MujocoRMGSim._run_sim_loop = original_loop

    sim = sim_holder.get("sim")
    metadata = {
        "video": str(args.video.resolve()),
        "runner": str(RUNNER),
        "runner_args": runner_args,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "camera": {
            "distance": args.distance,
            "azimuth": args.azimuth,
            "elevation": args.elevation,
        },
        **render_info,
    }
    if sim is not None:
        metadata["reference_duration_s"] = float(getattr(sim, "ref_duration_s", 0.0))
    metadata_path = args.video.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[sim2sim-video] wrote {args.video} and {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
