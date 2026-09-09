#!/usr/bin/env python3
"""
将 g1_23dof_rmg_aac（G123DoFRMGCfg）训练得到的 RMG Actor 导出为 ONNX。
仿照 save_cp_stu.py；默认 checkpoint 为 logs/g1_rmg_aac/g1_rmg_260403_113940/model_40000.pt。

说明：配置名 g1_23dof 对应 21 个驱动关节（腕部 roll 固定），与 checkpoint 中 num_actions=21 一致。
默认将 normalizer 打包进 ONNX（与训练 normalize_obs 一致），部署端可直接喂与训练相同的 raw obs 向量。
"""
import json
import os
import sys

_LEGGED_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.normpath(os.path.join(_LEGGED_ROOT, ".."))
sys.path.insert(0, os.path.join(_LEGGED_ROOT, "..", "rsl_rl"))
sys.path.insert(0, os.path.join(_LEGGED_ROOT, "..", "rmg_reference", "RMG_AAC", "rsl_rl"))

import numpy as np
import torch
import torch.nn as nn
import argparse

DEFAULT_MODEL_PATH = os.path.join(
    _REPO_ROOT,
    "logs",
    "g1_rmg_aac",
    "g1_rmg_260403_113940",
    "model_40000.pt",
)

RMG_ACTOR_KEYS = ("E_base", "E_hist", "G_ref", "G_ctx", "D_a")


def _get_activation(name):
    act_map = {
        "elu": nn.ELU, "selu": nn.SELU, "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "silu": nn.SiLU,
    }
    return act_map.get(name, nn.ELU)()


class ActorOnlyRMG(nn.Module):
    """仅包含 RMG actor 的轻量模块：E_base, E_hist, G_ref, G_ctx, D_a。"""

    def __init__(self, n_obs_single, history_len, num_actions,
                 base_hidden_dims=(512, 256), hist_hidden_dims=(512, 256),
                 ref_encoder_dims=(256, 128), ctx_encoder_dims=(128, 128),
                 action_decoder_dims=(256, 256), d_ref=12, d_ctx=12,
                 activation="elu", layer_norm=True):
        super().__init__()
        self.n_obs_single = n_obs_single
        self.history_len = history_len
        self.n_hist = history_len * n_obs_single
        self.dim_h = hist_hidden_dims[-1]
        act_fn = _get_activation(activation)

        layers = [nn.Linear(n_obs_single, base_hidden_dims[0]), act_fn]
        for i in range(len(base_hidden_dims) - 1):
            layers += [nn.Linear(base_hidden_dims[i], base_hidden_dims[i + 1]), act_fn]
        self.E_base = nn.Sequential(*layers)
        dim_f_base = base_hidden_dims[-1]

        layers = [nn.Linear(self.n_hist, hist_hidden_dims[0]), act_fn]
        for i in range(len(hist_hidden_dims) - 1):
            layers += [nn.Linear(hist_hidden_dims[i], hist_hidden_dims[i + 1]), act_fn]
        self.E_hist = nn.Sequential(*layers)

        dim_fusion = dim_f_base + hist_hidden_dims[-1]
        layers = [nn.Linear(dim_fusion, ref_encoder_dims[0]), act_fn]
        for i in range(len(ref_encoder_dims) - 1):
            layers += [nn.Linear(ref_encoder_dims[i], ref_encoder_dims[i + 1]), act_fn]
        layers.append(nn.Linear(ref_encoder_dims[-1], d_ref))
        self.G_ref = nn.Sequential(*layers)

        layers = [nn.Linear(dim_fusion, ctx_encoder_dims[0]), act_fn]
        for i in range(len(ctx_encoder_dims) - 1):
            layers += [nn.Linear(ctx_encoder_dims[i], ctx_encoder_dims[i + 1]), act_fn]
        layers.append(nn.Linear(ctx_encoder_dims[-1], d_ctx))
        self.G_ctx = nn.Sequential(*layers)

        dim_decoder_in = dim_f_base + d_ref + d_ctx
        layers = [nn.Linear(dim_decoder_in, action_decoder_dims[0]), act_fn]
        for i in range(len(action_decoder_dims) - 1):
            layers.append(nn.Linear(action_decoder_dims[i], action_decoder_dims[i + 1]))
            if layer_norm and i == len(action_decoder_dims) - 2:
                layers.append(nn.LayerNorm(action_decoder_dims[i + 1]))
            layers.append(act_fn)
        layers.append(nn.Linear(action_decoder_dims[-1], num_actions))
        self.D_a = nn.Sequential(*layers)

    def forward(self, obs):
        base_obs = obs[:, :self.n_obs_single]
        hist_flat = obs[:, self.n_obs_single:]
        f_base = self.E_base(base_obs)
        if self.n_hist > 0:
            h_t = self.E_hist(hist_flat)
        else:
            h_t = torch.zeros(obs.shape[0], self.dim_h, device=obs.device, dtype=obs.dtype)
        fusion = torch.cat([f_base, h_t], dim=-1)
        zhat_ref = self.G_ref(fusion)
        zhat_ctx = self.G_ctx(fusion)
        return self.D_a(torch.cat([f_base, zhat_ref, zhat_ctx], dim=-1))


class ObsNormActor(nn.Module):
    """将 normalizer + actor 打包，输入 raw obs，输出 actions。"""

    def __init__(self, actor, mean, std, eps=1e-4, clip=100.0):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.eps = eps
        self.clip = clip

    def forward(self, obs):
        x = (obs - self.mean) / (self.std + self.eps)
        x = torch.clamp(x, -self.clip, self.clip)
        return self.actor(x)


def infer_rmg_arch(state_dict):
    def s(key):
        t = state_dict.get(key)
        return t.shape if t is not None else None

    w = s("E_base.0.weight")
    if w is None:
        raise ValueError("state_dict 中缺少 E_base，无法推断 RMG 架构")
    n_obs_single = w[1]
    base_h = [w[0], (s("E_base.2.weight") or [0, 256])[0]]
    hist_w = s("E_hist.0.weight")
    n_hist = hist_w[1] if hist_w else 970
    hist_h = [hist_w[0] if hist_w else 512, (s("E_hist.2.weight") or [0, 256])[0]]
    d_ref = (s("G_ref.4.weight") or [12])[0]
    d_ctx = (s("G_ctx.4.weight") or [12])[0]
    num_actions = (s("D_a.5.weight") or [21])[0]
    return {
        "n_obs_single": n_obs_single,
        "history_len": n_hist // n_obs_single if n_obs_single > 0 else 10,
        "num_actions": num_actions,
        "base_hidden_dims": base_h,
        "hist_hidden_dims": hist_h,
        "ref_encoder_dims": [256, 128],
        "ctx_encoder_dims": [128, 128],
        "action_decoder_dims": [256, 256],
        "d_ref": d_ref,
        "d_ctx": d_ctx,
    }


def detect_model_type(state_dict):
    keys = set(state_dict.keys())
    if any(k.startswith("E_base") for k in keys):
        return "rmg"
    if any(k.startswith("actor.") for k in keys):
        return "mimic"
    raise ValueError("无法从 state_dict 推断模型类型，请显式指定 --model_type")


def _find_in_dict(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_in_dict(v, key)
            if r is not None:
                return r
    elif isinstance(d, (list, tuple)):
        for v in d:
            r = _find_in_dict(v, key)
            if r is not None:
                return r
    return None


def _infer_mimic_num_obs(model_path):
    cfg_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        val = _find_in_dict(cfg, "num_observations")
        return int(val) if val is not None else None
    except Exception:
        return None


def export_rmg_actor(args):
    device = torch.device("cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    actor_sd = {k: v for k, v in state_dict.items()
                if any(k.startswith(p) for p in RMG_ACTOR_KEYS)}
    arch = infer_rmg_arch(state_dict)
    print(f"推断架构: {arch}")

    actor = ActorOnlyRMG(**arch, activation=args.activation or "elu")
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()

    num_obs = arch["n_obs_single"] * (arch["history_len"] + 1)
    obs_input = torch.ones(1, num_obs, device=device)

    if "normalizer" in ckpt and args.with_normalizer:
        norm = ckpt["normalizer"]
        mean = norm._mean.detach().cpu().numpy()
        std = norm._std.detach().cpu().numpy()
        clip = getattr(norm, "_clip", np.inf)
        clip = float(clip) if clip != np.inf else 100.0
        eps = getattr(norm, "_eps", 1e-4)
        model = ObsNormActor(actor, mean, std, eps=eps, clip=clip)
        print(f"已包含 normalizer (clip={clip})，部署端可直接喂 raw obs")
    else:
        model = actor
        if "normalizer" in ckpt and not args.with_normalizer:
            print("提示: checkpoint 含 normalizer，可用 --with_normalizer 打包进 ONNX")

    out_dir = args.output_dir or os.path.dirname(args.model_path)
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    onnx_path = os.path.join(out_dir, f"{base_name}_actor.onnx")

    torch.onnx.export(
        model=model,
        args=obs_input,
        f=onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX to {os.path.abspath(onnx_path)}")
    return onnx_path


def export_mimic_actor(args):
    from rsl_rl.modules.actor_critic_rmg import Actor, get_activation

    device = torch.device("cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    actor_sd = {k.replace("actor.", ""): v for k, v in state_dict.items()
                if k.startswith("actor.")}
    if not actor_sd:
        raise ValueError("未找到任何 actor.* 参数")

    num_obs = args.num_obs
    if num_obs is None:
        num_obs = _infer_mimic_num_obs(args.model_path)
        if num_obs is not None:
            print(f"从 config.json 推断 num_obs={num_obs}")
    if num_obs is None:
        raise ValueError("mimic 模型需指定 --num_obs，或确保同目录存在 config.json 含 num_observations")
    num_motion_obs = args.num_motion_obs or (8 + 21)
    num_actions = args.num_actions or 21

    actor = Actor(
        num_observations=num_obs,
        num_motion_observations=num_motion_obs,
        num_motion_steps=1,
        motion_latent_dim=args.motion_latent_dim or 128,
        num_actions=num_actions,
        actor_hidden_dims=args.actor_hidden_dims or [1024, 1024, 512, 256],
        activation=get_activation(args.activation or "silu"),
        layer_norm=True,
        tanh_encoder_output=False,
    )
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()

    obs_input = torch.ones(1, num_obs, device=device)
    if "normalizer" in ckpt and args.with_normalizer:
        norm = ckpt["normalizer"]
        mean = norm._mean.detach().cpu().numpy()
        std = norm._std.detach().cpu().numpy()
        clip = getattr(norm, "_clip", np.inf)
        clip = float(clip) if clip != np.inf else 100.0
        eps = getattr(norm, "_eps", 1e-4)
        model = ObsNormActor(actor, mean, std, eps=eps, clip=clip)
        print(f"已包含 normalizer (clip={clip})")
    else:
        model = actor

    out_dir = args.output_dir or os.path.dirname(args.model_path)
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    onnx_path = os.path.join(out_dir, f"{base_name}_actor.onnx")

    torch.onnx.export(
        model=model,
        args=obs_input,
        f=onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX to {os.path.abspath(onnx_path)}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(
        description="g1_23dof RMG：将 actor 导出为 ONNX（默认使用 g1_rmg_260403_113940/model_40000.pt）"
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=DEFAULT_MODEL_PATH,
        help=f"checkpoint 路径（默认: {DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=["rmg", "mimic", "auto"], default="auto")
    parser.add_argument("--with_normalizer", action="store_true", default=True,
                        help="将 normalizer 打包进 ONNX（默认 True）")
    parser.add_argument("--no_normalizer", action="store_true", help="不包含 normalizer")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 版本")
    parser.add_argument("--activation", type=str, default=None)

    parser.add_argument("--num_obs", type=int, default=None)
    parser.add_argument("--num_motion_obs", type=int, default=None)
    parser.add_argument("--num_actions", type=int, default=21)
    parser.add_argument("--motion_latent_dim", type=int, default=128)
    parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=None)

    args = parser.parse_args()
    args.with_normalizer = getattr(args, "with_normalizer", True) and not getattr(args, "no_normalizer", False)

    if not os.path.isfile(args.model_path):
        print(f"错误: 找不到 checkpoint: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model_type = args.model_type
    if model_type == "auto":
        model_type = detect_model_type(state_dict)
        print(f"自动推断模型类型: {model_type}")

    if model_type == "rmg":
        export_rmg_actor(args)
    else:
        export_mimic_actor(args)


if __name__ == "__main__":
    main()
