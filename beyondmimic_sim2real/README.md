# BeyondMimic G1 23DoF sim2real-in-sim

DDS 闭环验证工具（策略 FSM 默认只允许 `--network lo`）。

## 文件

- `g1_beyondmimic_power_on.py` — **安全上电（无策略）**：零力矩 → 慢站 → 锁站姿 → 阻尼
- `g1_beyondmimic_fsm_sdk_lo.py` — 完整 FSM（站立 / 策略 / hold / 阻尼）+ 键盘/手柄
- `g1_beyondmimic_sdk_lo.py` — RobotIO、电机映射、观测对齐
- `deploy_mujocofor23.py` — ONNX metadata 与 policy↔xml 关节序
- `g1_play/` — `policy.onnx` + `motion.npz`
- `common/` — 手柄解析与可选全局键盘

## 1) 安全上电（无策略）

```bash
# 仿真
python g1_beyondmimic_power_on.py --network lo

# 真机（吊带+急停就绪）
python g1_beyondmimic_power_on.py --network enpXs0 --allow-real
```

流程：`ZERO_TORQUE` → Start/Space 慢站 → `READY_STAND` → Y/D 阻尼；Select/Q 退出。按 A 目前只提示策略未启用。

## 2) 仿真完整策略 FSM

先启动已打补丁的 `unitree_mujoco`（`use_joystick: 1` 时接 Xbox），再：

```bash
python g1_beyondmimic_fsm_sdk_lo.py --network lo
```
