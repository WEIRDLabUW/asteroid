import argparse
import csv
import os

import wandb
import numpy as np

from environments.procgen_env import make_maze_envs
from experiments.train_asteroid_procgen import evaluate_policy_on_envs_procgen


class ProcgenExpertPolicy:
    def __init__(self, env):
        self.env = env

    def reset(self, resets):
        del resets

    def update_context(self, states, actions, rewards, dones):
        del states, actions, rewards, dones

    def get_action(self, states):
        del states
        actions = np.zeros(self.env.n, dtype=np.int32)
        for i in range(self.env.n):
            opt = self.env._opt[i]
            pos = self.env._apos[i]
            if opt is None or pos is None:
                actions[i] = 0
                continue
            ax, ay = pos
            action = int(opt[ay, ax])
            actions[i] = action if action >= 0 else 0
        return actions


def write_metrics_csv(path, row):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate procgen maze expert.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-envs", type=int, default=16)
    parser.add_argument("--n-eval-envs", type=int, default=100)
    parser.add_argument("--visibility", type=int, default=3)
    parser.add_argument("--train-start-level", type=int, default=0)
    parser.add_argument("--train-num-levels", type=int, default=1000)
    parser.add_argument("--eval-start-level", type=int, default=1000)
    parser.add_argument("--eval-num-levels", type=int, default=1000)
    parser.add_argument("--eval-horizon", type=int, default=800)
    parser.add_argument("--env-interactions", type=int, default=0)
    parser.add_argument("--metrics-csv", type=str, default="results/expert_procgen/expert_seed0.csv")
    parser.add_argument("--save-dir", type=str, default="results/expert_procgen/eval_videos")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="asteroid-procgen")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.wandb_name or f"expert-procgen-seed{args.seed}",
        )

    _, eval_env = make_maze_envs(
        n_train=args.n_train_envs,
        n_eval=args.n_eval_envs,
        train_start=args.train_start_level,
        train_levels=args.train_num_levels,
        eval_start=args.eval_start_level,
        eval_levels=args.eval_num_levels,
        visibility=args.visibility,
        local_window_obs=True,
    )
    policy = ProcgenExpertPolicy(eval_env)
    stats = evaluate_policy_on_envs_procgen(
        eval_envs=eval_env,
        policy=policy,
        eval_horizon=args.eval_horizon,
        save_dir=args.save_dir,
        dagger_step="expert",
        eval_name="expert",
    )
    row = {"env_interactions": args.env_interactions, **stats}
    write_metrics_csv(args.metrics_csv, row)
    if args.log_wandb:
        payload = {
            "env_interactions": args.env_interactions,
            "eval/env_interactions": args.env_interactions,
        }
        payload.update({f"eval/expert_{k}": v for k, v in stats.items()})
        wandb.log(payload)
        wandb.finish()

    print(
        f"Expert | return={stats['mean_return']:.4f} | "
        f"success_rate={stats['success_rate']:.4f} | "
        f"mean_length={stats['mean_length']:.2f}"
    )


if __name__ == "__main__":
    main()
