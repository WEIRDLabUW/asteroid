import argparse
import csv
import os

import numpy as np
import torch
import tqdm
import wandb

from models import Transformer
from environments.procgen_env import make_maze_envs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_metrics_csv(path, row):
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class DPTProcgenPolicy:
    def __init__(self, model, batch_size, temp=0.0):
        self.model = model
        self.batch_size = batch_size
        self.temp = temp
        self.state_dim = model.config["state_dim"]
        self.action_dim = model.config["action_dim"]
        self.context_horizon = model.horizon
        self.zeros = torch.zeros(
            batch_size, self.state_dim ** 2 + self.action_dim + 1
        ).float().to(device)
        self.context_states = [[] for _ in range(batch_size)]
        self.context_actions = [[] for _ in range(batch_size)]
        self.context_next_states = [[] for _ in range(batch_size)]
        self.context_rewards = [[] for _ in range(batch_size)]

    def reset(self, resets):
        for i, reset in enumerate(resets):
            if reset:
                self.context_states[i] = []
                self.context_actions[i] = []
                self.context_next_states[i] = []
                self.context_rewards[i] = []

    def _build_batch(self, obs):
        flat_obs = obs.reshape(obs.shape[0], -1).astype(np.float32)
        max_len = min(
            max((len(hist) for hist in self.context_states), default=0),
            self.context_horizon,
        )

        context_states = torch.zeros(
            (self.batch_size, max_len, self.state_dim), dtype=torch.float32, device=device
        )
        context_actions = torch.zeros(
            (self.batch_size, max_len, self.action_dim), dtype=torch.float32, device=device
        )
        context_next_states = torch.zeros(
            (self.batch_size, max_len, self.state_dim), dtype=torch.float32, device=device
        )
        context_rewards = torch.zeros(
            (self.batch_size, max_len, 1), dtype=torch.float32, device=device
        )

        for i in range(self.batch_size):
            hist_len = min(len(self.context_states[i]), max_len)
            if hist_len == 0:
                continue
            context_states[i, :hist_len] = torch.tensor(
                np.array(self.context_states[i][-hist_len:]),
                dtype=torch.float32,
                device=device,
            )
            context_actions[i, :hist_len] = torch.tensor(
                np.array(self.context_actions[i][-hist_len:]),
                dtype=torch.float32,
                device=device,
            )
            context_next_states[i, :hist_len] = torch.tensor(
                np.array(self.context_next_states[i][-hist_len:]),
                dtype=torch.float32,
                device=device,
            )
            context_rewards[i, :hist_len, 0] = torch.tensor(
                np.array(self.context_rewards[i][-hist_len:]),
                dtype=torch.float32,
                device=device,
            )

        return {
            "query_states": torch.tensor(flat_obs, dtype=torch.float32, device=device),
            "context_states": context_states,
            "context_actions": context_actions,
            "context_next_states": context_next_states,
            "context_rewards": context_rewards,
            "zeros": self.zeros,
        }

    @torch.no_grad()
    def get_action(self, obs):
        self.model.eval()
        batch = self._build_batch(obs)
        logits = self.model(batch)
        if logits.ndim == 3:
            logits = logits[:, -1, :]

        if self.temp > 0:
            probs = torch.softmax(logits / self.temp, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
        return logits.argmax(dim=-1).cpu().numpy()

    def update_context(self, obs, actions, next_obs, rewards, dones):
        flat_obs = obs.reshape(obs.shape[0], -1).astype(np.float32)
        flat_next_obs = next_obs.reshape(next_obs.shape[0], -1).astype(np.float32)

        for i in range(self.batch_size):
            action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
            action_one_hot[int(actions[i])] = 1.0
            self.context_states[i].append(flat_obs[i])
            self.context_actions[i].append(action_one_hot)
            self.context_next_states[i].append(flat_next_obs[i])
            self.context_rewards[i].append(float(rewards[i]))


def evaluate_policy_on_envs_procgen_dpt(eval_envs, policy, eval_horizon):
    obs, _ = eval_envs.reset()
    resets = np.ones(eval_envs.n, dtype=bool)
    policy.reset(resets)

    n = eval_envs.n
    done_flag = np.zeros(n, dtype=bool)
    episode_rewards = np.zeros(n, dtype=np.float32)
    successes = np.zeros(n, dtype=bool)
    episode_lengths = np.zeros(n, dtype=int)

    pbar = tqdm.tqdm(range(eval_horizon), desc="Eval steps", unit="step")
    for _ in pbar:
        actions = policy.get_action(obs)
        next_obs, rewards, dones, infos = eval_envs.step(actions)
        policy.update_context(obs, actions, next_obs, rewards, dones)
        policy.reset(dones)
        obs = next_obs

        episode_lengths[~done_flag] += 1

        for i in range(n):
            if not done_flag[i]:
                episode_rewards[i] += rewards[i]
            if dones[i] and not done_flag[i]:
                done_flag[i] = True
                if infos[i].get("success", False):
                    successes[i] = True

        pbar.set_postfix(
            completed=int(done_flag.sum()),
            success_rate=f"{np.mean(successes):.3f}",
        )

        if done_flag.all():
            break

    return {
        "mean_return": float(np.mean(episode_rewards)),
        "std_return": float(np.std(episode_rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_success_length": float(np.mean(episode_lengths[successes])) if successes.any() else 0.0,
        "completed_episodes": int(done_flag.sum()),
        "total_episodes": int(n),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate DPT checkpoint on procgen maze.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--eval-horizon", type=int, default=800)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--procgen-train-envs", type=int, default=16)
    parser.add_argument("--procgen-eval-envs", type=int, default=100)
    parser.add_argument("--procgen-train-start", type=int, default=0)
    parser.add_argument("--procgen-train-levels", type=int, default=1000)
    parser.add_argument("--procgen-eval-start", type=int, default=1000)
    parser.add_argument("--procgen-eval-levels", type=int, default=1000)
    parser.add_argument("--procgen-visibility", type=int, default=3)
    parser.add_argument("--env-interactions", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="asteroid-procgen")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.wandb_name or f"dpt-procgen-eval-seed{args.seed}",
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _, eval_env = make_maze_envs(
        n_train=args.procgen_train_envs,
        n_eval=args.procgen_eval_envs,
        train_start=args.procgen_train_start,
        train_levels=args.procgen_train_levels,
        eval_start=args.procgen_eval_start,
        eval_levels=args.procgen_eval_levels,
        visibility=args.procgen_visibility,
        local_window_obs=True,
    )

    obs, _ = eval_env.reset()
    state_dim = int(np.prod(obs.shape[1:]))
    action_dim = eval_env.action_space.n

    cfg = {
        "horizon": args.horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "shuffle": False,
        "dropout": args.dropout,
        "test": True,
        "store_gpu": True,
        "continuous_action": False,
        "rollin_type": "uniform",
    }

    model = Transformer(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    policy = DPTProcgenPolicy(
        model=model,
        batch_size=eval_env.n,
        temp=args.temp,
    )
    stats = evaluate_policy_on_envs_procgen_dpt(
        eval_envs=eval_env,
        policy=policy,
        eval_horizon=args.eval_horizon,
    )

    print(
        f"Evaluated {stats['completed_episodes']}/{stats['total_episodes']} envs | "
        f"mean_return={stats['mean_return']:.4f} | "
        f"std_return={stats['std_return']:.4f} | "
        f"success_rate={stats['success_rate']:.4f} | "
        f"mean_length={stats['mean_length']:.2f}"
    )
    if args.log_wandb:
        payload = {
            "env_interactions": args.env_interactions,
            "eval/env_interactions": args.env_interactions,
        }
        payload.update({f"eval/{k}": v for k, v in stats.items()})
        wandb.log(payload)
        wandb.finish()

    append_metrics_csv(
        args.metrics_csv,
        {
            "env_interactions": args.env_interactions,
            **stats,
        },
    )


if __name__ == "__main__":
    main()
