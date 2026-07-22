#!/usr/bin/env bash
# Procgen maze benchmark: expert eval, BC+PPO baseline, and ASTEROID (procgen).
# Requires the procgen extra: `uv sync --extra procgen`.
#   SEEDS="0 1 2" bash scripts/run_procgen.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
VISIBILITY="${VISIBILITY:-3}"
WANDB="${WANDB:-0}"
FLAGS=(); [[ "$WANDB" == "1" ]] && FLAGS=(--log_wandb)
DASH=();  [[ "$WANDB" == "1" ]] && DASH=(--log-wandb)

for seed in $SEEDS; do
  echo "=== procgen expert eval | seed $seed ==="
  uv run python experiments/eval_procgen_expert.py --seed "$seed" --visibility "$VISIBILITY" "${DASH[@]}"

  echo "=== procgen ASTEROID | seed $seed ==="
  uv run python experiments/train_asteroid_procgen.py --seed "$seed" --visibility "$VISIBILITY" "${FLAGS[@]}"

  echo "=== procgen BC+PPO | seed $seed ==="
  uv run python experiments/train_procgen_bc_ppo.py --seed "$seed" --visibility "$VISIBILITY" "${FLAGS[@]}"
done
