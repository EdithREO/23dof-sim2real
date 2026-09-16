#!/usr/bin/env bash
# Overlay this repo's unitree_mujoco patches onto a full upstream checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${1:-$ROOT/unitree_mujoco}"

if [[ ! -d "$DST/simulate" || ! -d "$DST/unitree_robots/g1" ]]; then
  echo "error: '$DST' does not look like a unitree_mujoco tree." >&2
  echo "clone upstream first, e.g.:" >&2
  echo "  git clone https://github.com/unitreerobotics/unitree_mujoco.git \"$DST\"" >&2
  exit 1
fi

if [[ ! -d "$DST/unitree_robots/g1/meshes" ]]; then
  echo "error: missing meshes under $DST/unitree_robots/g1/meshes" >&2
  echo "use a full upstream clone; this repo only ships XML patches." >&2
  exit 1
fi

copy() {
  local rel="$1"
  local src="$ROOT/$rel"
  local dst="$DST/${rel#unitree_mujoco/}"
  if [[ ! -f "$src" ]]; then
    echo "error: missing patch source $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "applied $rel"
}

copy unitree_mujoco/simulate/src/main.cc
copy unitree_mujoco/simulate/src/unitree_sdk2_bridge.h
copy unitree_mujoco/simulate/config.yaml
copy unitree_mujoco/unitree_robots/g1/g1_23dof.xml
copy unitree_mujoco/unitree_robots/g1/scene_23dof.xml

# Keep configs/ mirror in sync for documentation copies.
if [[ -f "$ROOT/configs/unitree_mujoco.simulate.config.yaml" ]]; then
  cp -a "$ROOT/unitree_mujoco/simulate/config.yaml" \
    "$ROOT/configs/unitree_mujoco.simulate.config.yaml"
fi

echo "done. Next: link MuJoCo under simulate/mujoco and build simulate/build."
