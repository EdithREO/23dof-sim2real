# BeyondMimic G1 23DoF sim2real-in-sim

DDS/`lo` 闭环验证（非真机程序）：

- `g1_beyondmimic_fsm_sdk_lo.py` — FSM（站立 / 策略 / hold / 阻尼）+ 键盘/手柄
- `g1_beyondmimic_sdk_lo.py` — RobotIO、电机映射、观测对齐
- `deploy_mujocofor23.py` — ONNX metadata 与 policy↔xml 关节序
- `g1_play/` — `policy.onnx` + `motion.npz`
- `common/` — 手柄解析与可选全局键盘

先启动已打补丁的 `unitree_mujoco`（`use_joystick: 1` 时接 Xbox），再：

```bash
python g1_beyondmimic_fsm_sdk_lo.py --network lo
```
