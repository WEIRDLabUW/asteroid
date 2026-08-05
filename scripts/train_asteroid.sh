#!/usr/bin/env bash
# Train ASTEROID on the gridworld / navigation environments.
# Override the environment/seed sweep from the shell, e.g.
#   ENVS="darkroom-easy keydoor-markovian" SEEDS="0 1 2" bash scripts/train_asteroid.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENVS="${ENVS:-darkroom-easy darkroom-hard keydoor-markovian keydoor-nonmarkovian}"
SEEDS="${SEEDS:-0 1 2}"
WANDB="${WANDB:-0}"
FLAGS=(); [[ "$WANDB" == "1" ]] && FLAGS=(--log_wandb)

for env in $ENVS; do
  for seed in $SEEDS; do
    echo "=== ASTEROID | $env | seed $seed ==="
    uv run python experiments/train_asteroid.py \
      --config "configs/${env}.yaml" --seed "$seed" "${FLAGS[@]}"
  done
done
