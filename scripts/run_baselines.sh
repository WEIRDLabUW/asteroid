#!/usr/bin/env bash
# Train the gridworld baselines (DPT, RL2, AAWR) on the same env/seed sweep.
# RL2 needs the baselines extra: `uv sync --extra baselines`.
#   ENVS="darkroom-easy" SEEDS="0" METHODS="dpt aawr" bash scripts/run_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENVS="${ENVS:-darkroom-easy darkroom-hard keydoor-markovian keydoor-nonmarkovian}"
SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-dpt rl2 aawr}"
WANDB="${WANDB:-0}"
FLAGS=(); [[ "$WANDB" == "1" ]] && FLAGS=(--log_wandb)

for method in $METHODS; do
  for env in $ENVS; do
    for seed in $SEEDS; do
      echo "=== ${method} | $env | seed $seed ==="
      uv run python "experiments/train_${method}.py" \
        --config "configs/${env}.yaml" --seed "$seed" "${FLAGS[@]}"
    done
  done
done
