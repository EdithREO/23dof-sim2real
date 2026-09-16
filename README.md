# 23DoF G1 sim2real

Unitree G1（货架 23DoF 机型）的 sim2sim / SDK sim2real-in-sim 工作区。

**新机器请先读 [SETUP.md](SETUP.md)**（依赖、编译、打补丁、跑 FSM）。本 README 只作速览。

包含两条策略路径：

- **RMG `model_24500`**：`mage_round2_model24500_sim2sim_bundle/`（21 主动关节）
- **BeyondMimic Tracking-Flat-G123**：`beyondmimic_sim2real/`（23 主动关节，FSM + 手柄）

## 第三方仓库（请单独 clone）

本仓库不嵌入上游完整历史与编译产物。在同级目录执行：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
git clone https://github.com/unitreerobotics/unitree_sdk2.git
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git
```

然后执行：

```bash
./scripts/apply_unitree_mujoco_patches.sh
```

补丁说明见 [SETUP.md](SETUP.md)。也可把 `configs/unitree_mujoco.simulate.config.yaml` 拷到 `unitree_mujoco/simulate/config.yaml`。

## BeyondMimic FSM sim2real-in-sim（推荐演示）

完整步骤见 SETUP。摘要：

```bash
# 终端 1
export DISPLAY=:1
./unitree_mujoco/simulate/build/unitree_mujoco

# 终端 2
conda activate unitree_rl_mjlab
export CYCLONEDDS_HOME=$PWD/cyclonedds/install
cd beyondmimic_sim2real
python g1_beyondmimic_fsm_sdk_lo.py --network lo
```

| 功能 | 键盘 | 手柄（Xbox） |
|---|---|---|
| 站立 / 停策略 | Space | Start |
| 策略 | P | A |
| Hold | H | B |
| 阻尼 | D | Y |
| 退出 | Q / Esc | Select |

无手柄时把 `use_joystick` 改回 `0`。

## RMG Python sim2sim

```bash
conda activate unitree_rl_mjlab
cd mage_round2_model24500_sim2sim_bundle
python tw_g1_mujoco_rmg/scripts/g1_mujoco_sim_rmg_onnx_csv.py \
  --xml assets/unitree_g1_23dof/g1_21dof.xml \
  --onnx policy/model_24500_actor.onnx \
  --motion-csv motion/neutral_held_50hz.csv \
  --csv-fps 50 --device cpu
```

## RMG SDK 闭环（unitree_mujoco / `lo`）

```bash
export CYCLONEDDS_HOME=/path/to/cyclonedds/install
python mage_round2_model24500_sim2sim_bundle/tools/g1_23dof_sdk_sim2real.py \
  --network lo --command-mode position --duration 20
```

真机需要 `--network <网卡> --allow-real`，并进入 debug、关掉机载运控。
