# Setup：另一台机器复现 23DoF G1 sim2real-in-sim

本仓库**不是**完整可执行镜像。GitHub 上有：

- BeyondMimic FSM/SDK 脚本 + `policy.onnx` / `motion.npz`
- RMG `model_24500` sim2sim bundle
- `unitree_mujoco` **补丁文件**（不是完整上游树、不含 `meshes/`、不含 `build/`）

需要自行准备：系统依赖、MuJoCo、unitree_sdk2、cyclonedds、完整 `unitree_mujoco`、Python 环境。

下面以 Linux x86_64 + conda 为例。默认工作目录：

```bash
export WORK=~/Project/23dof_sim2real
```

---

## 0. 系统依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev libfmt-dev \
  joystick
```

可选：把用户加入 `input` 组以便读手柄（世界可读时也可不改）：

```bash
sudo usermod -aG input "$USER"
# 重新登录后生效
```

---

## 1. 克隆本仓库

```bash
mkdir -p ~/Project && cd ~/Project
git clone https://github.com/EdithREO/23dof-sim2real.git
cd 23dof-sim2real
export WORK="$PWD"
```

---

## 2. 克隆并安装 unitree_sdk2

官方推荐装到 `/opt/unitree_robotics`（`unitree_mujoco` 的 CMake 会找这里）：

```bash
cd "$WORK"
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
sudo make install -j"$(nproc)"
```

---

## 3. 编译 CycloneDDS（Python SDK 需要）

```bash
cd "$WORK"
git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$WORK/cyclonedds/install"
cmake --build . -j"$(nproc)"
cmake --install .
export CYCLONEDDS_HOME="$WORK/cyclonedds/install"
```

把下面两行写进 `~/.bashrc`（或每次开终端执行）：

```bash
export CYCLONEDDS_HOME=~/Project/23dof_sim2real/cyclonedds/install
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}
```

---

## 4. 安装 unitree_sdk2_python

```bash
cd "$WORK"
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
# 使用与本机一致的 conda 环境后再 pip
pip install -e .
```

---

## 5. 准备完整 unitree_mujoco 并打上本仓库补丁

本仓库只跟踪补丁，**必须先 clone 上游**，再覆盖补丁文件：

```bash
cd "$WORK"

# 若目录已存在且带上游 .git，可直接用；否则：
if [ ! -d unitree_mujoco/.git ]; then
  git clone https://github.com/unitreerobotics/unitree_mujoco.git unitree_mujoco
fi

# 用仓库里的补丁覆盖上游对应文件
./scripts/apply_unitree_mujoco_patches.sh
```

补丁覆盖这些路径：

- `unitree_mujoco/simulate/src/main.cc`
- `unitree_mujoco/simulate/src/unitree_sdk2_bridge.h`
- `unitree_mujoco/simulate/config.yaml`
- `unitree_mujoco/unitree_robots/g1/g1_23dof.xml`
- `unitree_mujoco/unitree_robots/g1/scene_23dof.xml`

`meshes/` 必须来自上游 clone，仓库不包含网格。

### 5.1 MuJoCo C++ 库（给 simulate 链接）

从 [MuJoCo Releases](https://github.com/google-deepmind/mujoco/releases) 下载与系统匹配的包（本机验证过 3.3.x / 3.5.x 一类），解压到例如：

```bash
mkdir -p ~/.mujoco
# 假设解压得到 ~/.mujoco/mujoco-3.3.6
cd "$WORK/unitree_mujoco/simulate"
ln -sfn ~/.mujoco/mujoco-3.3.6 mujoco
```

### 5.2 编译仿真器

```bash
cd "$WORK/unitree_mujoco/simulate"
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
# 产物：./unitree_mujoco
```

---

## 6. Python / conda 环境

本机使用 conda 环境名 `unitree_rl_mjlab`。最小依赖：

```bash
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab
pip install mujoco onnxruntime numpy pandas pyyaml onnx pynput
# 再安装 unitree_sdk2_python（见第 4 步），并保证 CYCLONEDDS_HOME 已设置
```

检查：

```bash
python -c "import mujoco, onnxruntime, unitree_sdk2py; print('ok', mujoco.__version__)"
```

---

## 7. 跑 BeyondMimic FSM（sim2real-in-sim）

**终端 1 — 仿真（需要显示器，例如 `DISPLAY=:1`）：**

```bash
conda activate unitree_rl_mjlab   # 非必须，二进制不依赖 conda
export DISPLAY=:1                # 按你的桌面改
cd "$WORK/unitree_mujoco/simulate/build"
./unitree_mujoco
```

应看到 G1 `scene_23dof`，并打印类似：

- `BeyondMimic dynamics: ...`
- `paused; waiting for first rt/lowcmd`

**终端 2 — FSM 控制器：**

```bash
conda activate unitree_rl_mjlab
export CYCLONEDDS_HOME="$WORK/cyclonedds/install"
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}
cd "$WORK/beyondmimic_sim2real"
python g1_beyondmimic_fsm_sdk_lo.py --network lo
```

默认流程：`STAND_UP` → `READY_STAND`（hold），再手动进策略。

| 功能 | 键盘 | Xbox 手柄 |
|---|---|---|
| 站立 / 请求停策略 | Space | Start |
| 策略 | P | A |
| Hold | H | B |
| 阻尼 | D | Y |
| 退出 | Q / Esc | Select |

手柄：`config.yaml` 里 `use_joystick: 1`，设备默认 `/dev/input/js0`。  
**没有手柄时务必改成 `use_joystick: 0`**，否则仿真可能因打不开手柄直接退出。

---

## 8. 跑 RMG Python sim2sim（不依赖 unitree_mujoco）

```bash
conda activate unitree_rl_mjlab
cd "$WORK/mage_round2_model24500_sim2sim_bundle"
python tw_g1_mujoco_rmg/scripts/g1_mujoco_sim_rmg_onnx_csv.py \
  --xml assets/unitree_g1_23dof/g1_21dof.xml \
  --onnx policy/model_24500_actor.onnx \
  --motion-csv motion/neutral_held_50hz.csv \
  --csv-fps 50 --device cpu
```

---

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `DDS timeout: lowstate=... sport=...` | 先开 `unitree_mujoco`；确认 `interface: lo`、`domain_id: 0` |
| `Joystick open failed` / 仿真秒退 | `use_joystick: 0`，或修好 `/dev/input/js0` |
| `import unitree_sdk2py` 失败 | 设置 `CYCLONEDDS_HOME` 后重装 `unitree_sdk2_python` |
| 找不到 mesh | 确认上游 `unitree_mujoco/unitree_robots/g1/meshes` 存在，且补丁只覆盖 xml |
| 编译找不到 `unitree_sdk2` | 确认 `/opt/unitree_robotics` 安装成功 |
| 无 GUI | 设置正确 `DISPLAY` / `XAUTHORITY`，或用有桌面的机器 |

---

## 10. 目录期望结构（复现后）

```text
23dof_sim2real/
  beyondmimic_sim2real/          # 本仓库已有
  mage_round2_model24500_.../    # 本仓库已有
  configs/                       # 本仓库已有
  scripts/apply_unitree_mujoco_patches.sh
  SETUP.md
  unitree_mujoco/                # 上游 clone + 补丁覆盖 + build/
  unitree_sdk2/                  # 上游 clone（可装到 /opt）
  unitree_sdk2_python/           # 上游 clone + pip -e
  cyclonedds/install/            # 本地编译产物
```

真机部署不在本 SETUP 范围内：FSM 脚本强制 `--network lo`。真机需单独安全审查与 `--allow-real` 类流程。
