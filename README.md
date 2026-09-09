# 23DoF G1 sim2real

Unitree G1（货架 23DoF 机型）上的 RMG `model_24500` 策略 sim2sim / SDK 闭环工作区。

策略本身是 **21 个主动关节**（腕 roll 不进 ONNX），动作与评测脚本在：

`mage_round2_model24500_sim2sim_bundle/`

## 第三方仓库（请单独 clone）

本仓库不嵌入下面这些上游完整历史与编译产物。在同级目录执行：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
git clone https://github.com/unitreerobotics/unitree_sdk2.git
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git
```

`unitree_mujoco` 仿真对接本策略时，把 `configs/unitree_mujoco.simulate.config.yaml` 拷到 `unitree_mujoco/simulate/config.yaml`（G1、`scene_23dof.xml`、DDS `lo`、吊带默认打开）。

## Python sim2sim

```bash
conda activate unitree_rl_mjlab
cd mage_round2_model24500_sim2sim_bundle
python tw_g1_mujoco_rmg/scripts/g1_mujoco_sim_rmg_onnx_csv.py \
  --xml assets/unitree_g1_23dof/g1_21dof.xml \
  --onnx policy/model_24500_actor.onnx \
  --motion-csv motion/neutral_held_50hz.csv \
  --csv-fps 50 --device cpu
```

## SDK 闭环（先 unitree_mujoco / `lo`）

```bash
export CYCLONEDDS_HOME=/path/to/cyclonedds/install
python mage_round2_model24500_sim2sim_bundle/tools/g1_23dof_sdk_sim2real.py \
  --network lo --command-mode position --duration 20
```

真机需要 `--network <网卡> --allow-real`，并进入 debug、关掉机载运控。
