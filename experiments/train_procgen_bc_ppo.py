"""
BC + PPO baseline for Procgen Maze.

The policy is recurrent and receives the current partial observation plus the
previous action, reward, and done bit. It is warm-started with behavior cloning
from the procgen expert exposed by VecProcgenMaze infos["opt_action"], then
fine-tuned with clipped PPO. Rewards are -1 per environment step, making episode
return the negative number of steps to the goal.
"""

import argparse
import csv
import json
import os
import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import tqdm
import wandb

from environments.procgen_env import make_maze_envs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecurrentBCPPOPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=128, obs_dim=128):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.action_dim = action_dim
        flat_obs_dim = int(np.prod(obs_shape))
        self.obs_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_obs_dim, obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, obs_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=obs_dim + action_dim + 2,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observations, prev_actions, prev_rewards, prev_dones, hidden=None):
        bsz, seq_len = observations.shape[:2]
        obs = observations.reshape(bsz * seq_len, *self.obs_shape)
        obs_emb = self.obs_encoder(obs).reshape(bsz, seq_len, -1)
        prev_action_oh = F.one_hot(prev_actions.clamp_min(0), self.action_dim).float()
        x = torch.cat(
            [obs_emb, prev_action_oh, prev_rewards.unsqueeze(-1), prev_dones.unsqueeze(-1)],
            dim=-1,
        )
        out, hidden = self.gru(x, hidden)
        logits = self.action_head(out)
        values = self.value_head(out).squeeze(-1)
        return logits, values, hidden

    @torch.no_grad()
    def act(self, obs, prev_actions, prev_rewards, prev_dones, hidden=None, deterministic=True):
        self.eval()
        obs_t = torch.as_tensor(obs[:, None], dtype=torch.float32, device=DEVICE)
        pa_t = torch.as_tensor(prev_actions[:, None], dtype=torch.long, device=DEVICE)
        pr_t = torch.as_tensor(prev_rewards[:, None], dtype=torch.float32, device=DEVICE)
        pd_t = torch.as_tensor(prev_dones[:, None], dtype=torch.float32, device=DEVICE)
        logits, _, hidden = self.forward(obs_t, pa_t, pr_t, pd_t, hidden)
        logits = logits[:, -1]
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        return actions.cpu().numpy(), hidden


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        actions = np.asarray(traj["actions"], dtype=np.int64)
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        dones = np.asarray(traj["dones"], dtype=np.float32)
        prev_actions = np.concatenate([[0], actions[:-1]])
        prev_rewards = np.concatenate([[0.0], rewards[:-1]])
        prev_dones = np.concatenate([[1.0], dones[:-1]])
        return {
            "observations": torch.as_tensor(np.asarray(traj["observations"]), dtype=torch.float32),
            "actions": torch.as_tensor(actions, dtype=torch.long),
            "prev_actions": torch.as_tensor(prev_actions, dtype=torch.long),
            "prev_rewards": torch.as_tensor(prev_rewards, dtype=torch.float32),
            "prev_dones": torch.as_tensor(prev_dones, dtype=torch.float32),
        }


def collate_trajectories(batch):
    out = {}
    for key in batch[0]:
        out[key] = pad_sequence([item[key] for item in batch], batch_first=True)
    lengths = torch.as_tensor([item["actions"].shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max())
    mask = torch.arange(max_len)[None] < lengths[:, None]
    out["mask"] = mask.float()
    return out


def _new_traj():
    return {"observations": [], "actions": [], "rewards": [], "dones": [], "success": False}


def collect_expert_trajectories(env, target_interactions, env_horizon):
    obs, infos = env.reset()
    current = [_new_traj() for _ in range(env.n)]
    episode_lengths = np.zeros(env.n, dtype=np.int32)
    trajectories = []
    interactions = 0

    pbar = tqdm.tqdm(total=target_interactions, desc="Collecting BC expert data", unit="step")
    while interactions < target_interactions:
        actions = np.asarray([info.get("opt_action", 0) for info in infos], dtype=np.int32)
        next_obs, rewards, dones, next_infos = env.step(actions)
        interactions += env.n
        pbar.update(min(env.n, target_interactions - pbar.n))
        episode_lengths += 1
        dones = dones | (episode_lengths >= env_horizon)

        for i in range(env.n):
            current[i]["observations"].append(obs[i].copy())
            current[i]["actions"].append(int(actions[i]))
            current[i]["rewards"].append(float(rewards[i]))
            current[i]["dones"].append(bool(dones[i]))
            current[i]["success"] = current[i]["success"] or bool(next_infos[i].get("success", False))
            if dones[i]:
                trajectories.append(current[i])
                current[i] = _new_traj()
                episode_lengths[i] = 0

        obs, infos = next_obs, next_infos

    pbar.close()
    for traj in current:
        if len(traj["actions"]) > 0:
            trajectories.append(traj)
    return trajectories, interactions


def train_epochs(model, optimizer, trajectories, batch_size, epochs, max_grad_norm):
    dataset = TrajectoryDataset(trajectories)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
    )
    losses = []
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits, _, _ = model(
                batch["observations"],
                batch["prev_actions"],
                batch["prev_rewards"],
                batch["prev_dones"],
            )
            token_loss = F.cross_entropy(
                logits.reshape(-1, model.action_dim),
                batch["actions"].reshape(-1),
                reduction="none",
            ).reshape_as(batch["actions"])
            loss = (token_loss * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def collect_ppo_rollout(model, env, rollout_steps, env_horizon, gamma):
    model.eval()
    obs, _ = env.reset()
    n = env.n
    hidden = None
    prev_actions = np.zeros(n, dtype=np.int64)
    prev_rewards = np.zeros(n, dtype=np.float32)
    prev_dones = np.ones(n, dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    rows = {
        "observations": [], "prev_actions": [], "prev_rewards": [], "prev_dones": [],
        "actions": [], "rewards": [], "dones": [], "old_log_probs": [], "values": [],
    }

    for _ in tqdm.tqdm(range(rollout_steps), desc="Collecting PPO rollouts", unit="step"):
        obs_t = torch.as_tensor(obs[:, None], dtype=torch.float32, device=DEVICE)
        pa_t = torch.as_tensor(prev_actions[:, None], dtype=torch.long, device=DEVICE)
        pr_t = torch.as_tensor(prev_rewards[:, None], dtype=torch.float32, device=DEVICE)
        pd_t = torch.as_tensor(prev_dones[:, None], dtype=torch.float32, device=DEVICE)
        logits, values, hidden = model(obs_t, pa_t, pr_t, pd_t, hidden)
        logits = logits[:, -1]
        values = values[:, -1]
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        next_obs, rewards, dones, _ = env.step(actions.cpu().numpy())
        lengths += 1
        dones = dones | (lengths >= env_horizon)

        rows["observations"].append(obs.copy())
        rows["prev_actions"].append(prev_actions.copy())
        rows["prev_rewards"].append(prev_rewards.copy())
        rows["prev_dones"].append(prev_dones.copy())
        rows["actions"].append(actions.cpu().numpy())
        rows["rewards"].append(rewards.copy())
        rows["dones"].append(dones.copy())
        rows["old_log_probs"].append(log_probs.cpu().numpy())
        rows["values"].append(values.cpu().numpy())

        prev_actions = actions.cpu().numpy().astype(np.int64)
        prev_rewards = rewards.astype(np.float32)
        prev_dones = dones.astype(np.float32)
        obs = next_obs

        if hidden is not None and dones.any():
            done_idx = torch.as_tensor(dones, dtype=torch.bool, device=hidden.device)
            hidden[:, done_idx, :] = 0.0
        lengths[dones] = 0

    rollout = {k: np.asarray(v) for k, v in rows.items()}
    returns = np.zeros_like(rollout["rewards"], dtype=np.float32)
    running = np.zeros(n, dtype=np.float32)
    for t in reversed(range(rollout_steps)):
        running = rollout["rewards"][t] + gamma * running * (1.0 - rollout["dones"][t].astype(np.float32))
        returns[t] = running
    rollout["returns"] = returns
    rollout["advantages"] = returns - rollout["values"].astype(np.float32)
    adv = rollout["advantages"]
    rollout["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
    return rollout, int(rollout_steps * n)


def train_ppo_epochs(model, optimizer, rollout, epochs, clip_range, value_coef, entropy_coef, max_grad_norm):
    obs = torch.as_tensor(np.swapaxes(rollout["observations"], 0, 1), dtype=torch.float32, device=DEVICE)
    prev_actions = torch.as_tensor(np.swapaxes(rollout["prev_actions"], 0, 1), dtype=torch.long, device=DEVICE)
    prev_rewards = torch.as_tensor(np.swapaxes(rollout["prev_rewards"], 0, 1), dtype=torch.float32, device=DEVICE)
    prev_dones = torch.as_tensor(np.swapaxes(rollout["prev_dones"], 0, 1), dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(np.swapaxes(rollout["actions"], 0, 1), dtype=torch.long, device=DEVICE)
    old_log_probs = torch.as_tensor(np.swapaxes(rollout["old_log_probs"], 0, 1), dtype=torch.float32, device=DEVICE)
    returns = torch.as_tensor(np.swapaxes(rollout["returns"], 0, 1), dtype=torch.float32, device=DEVICE)
    advantages = torch.as_tensor(np.swapaxes(rollout["advantages"], 0, 1), dtype=torch.float32, device=DEVICE)

    losses = []
    model.train()
    for _ in range(epochs):
        logits, values, _ = model(obs, prev_actions, prev_rewards, prev_dones)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        pg_loss_1 = -advantages * ratio
        pg_loss_2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, env, eval_horizon):
    obs, _ = env.reset()
    n = env.n
    hidden = None
    prev_actions = np.zeros(n, dtype=np.int64)
    prev_rewards = np.zeros(n, dtype=np.float32)
    prev_dones = np.ones(n, dtype=np.float32)
    done_flag = np.zeros(n, dtype=bool)
    successes = np.zeros(n, dtype=bool)
    returns = np.zeros(n, dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)

    for _ in tqdm.tqdm(range(eval_horizon), desc="Evaluating BC+PPO", unit="step"):
        actions, hidden = model.act(obs, prev_actions, prev_rewards, prev_dones, hidden)
        next_obs, rewards, dones, infos = env.step(actions)
        active = ~done_flag
        returns[active] += rewards[active]
        lengths[active] += 1
        dones = dones | (lengths >= eval_horizon)

        for i in range(n):
            if active[i] and dones[i]:
                done_flag[i] = True
                successes[i] = bool(infos[i].get("success", False))

        prev_actions = actions.astype(np.int64)
        prev_rewards = rewards.astype(np.float32)
        prev_dones = dones.astype(np.float32)
        obs = next_obs

        if hidden is not None and dones.any():
            done_idx = torch.as_tensor(dones, dtype=torch.bool, device=hidden.device)
            hidden[:, done_idx, :] = 0.0
        if done_flag.all():
            break

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "mean_success_length": float(np.mean(lengths[successes])) if successes.any() else 0.0,
        "completed_episodes": int(done_flag.sum()),
        "total_episodes": int(n),
    }


def append_metrics_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(description="BC + PPO baseline for procgen maze.")
    parser.add_argument("--exp_name", type=str, default="bc_ppo_procgen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_env_interactions", type=int, default=200000)
    parser.add_argument("--log_interval_interactions", type=int, default=20000)
    parser.add_argument("--n_train_envs", type=int, default=16)
    parser.add_argument("--n_eval_envs", type=int, default=100)
    parser.add_argument("--visibility", type=int, default=3)
    parser.add_argument("--train_start_level", type=int, default=0)
    parser.add_argument("--train_num_levels", type=int, default=1000)
    parser.add_argument("--eval_start_level", type=int, default=1000)
    parser.add_argument("--eval_num_levels", type=int, default=1000)
    parser.add_argument("--env_horizon", type=int, default=800)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--obs_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs_per_iter", type=int, default=3)
    parser.add_argument("--ppo_steps_per_iter", type=int, default=2048)
    parser.add_argument("--ppo_epochs_per_iter", type=int, default=4)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--ppo_value_coef", type=float, default=0.5)
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_buffer_trajectories", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="results/bc_ppo_procgen")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="asteroid-procgen")
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = os.path.join(args.save_dir, f"{args.exp_name}-seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-seed{args.seed}",
        )

    train_env, eval_env = make_maze_envs(
        n_train=args.n_train_envs,
        n_eval=args.n_eval_envs,
        train_start=args.train_start_level,
        train_levels=args.train_num_levels,
        eval_start=args.eval_start_level,
        eval_levels=args.eval_num_levels,
        visibility=args.visibility,
        local_window_obs=True,
    )
    model = RecurrentBCPPOPolicy(
        obs_shape=train_env.observation_space.shape,
        action_dim=train_env.action_space.n,
        hidden_dim=args.hidden_dim,
        obs_dim=args.obs_dim,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = deque(maxlen=args.max_buffer_trajectories)

    total_interactions = 0
    metrics_path = os.path.join(run_dir, "metrics.csv")
    eval_results = []

    while total_interactions < args.total_env_interactions:
        to_collect = min(
            args.log_interval_interactions,
            args.total_env_interactions - total_interactions,
        )
        new_trajs, used_interactions = collect_expert_trajectories(
            train_env,
            target_interactions=to_collect,
            env_horizon=args.env_horizon,
        )
        total_interactions += used_interactions
        buffer.extend(new_trajs)

        train_loss = train_epochs(
            model,
            optimizer,
            list(buffer),
            batch_size=args.batch_size,
            epochs=args.train_epochs_per_iter,
            max_grad_norm=args.max_grad_norm,
        )
        ppo_loss = 0.0
        if args.ppo_steps_per_iter > 0 and args.ppo_epochs_per_iter > 0:
            rollout, ppo_interactions = collect_ppo_rollout(
                model,
                train_env,
                rollout_steps=args.ppo_steps_per_iter,
                env_horizon=args.env_horizon,
                gamma=args.gamma,
            )
            total_interactions += ppo_interactions
            ppo_loss = train_ppo_epochs(
                model,
                optimizer,
                rollout,
                epochs=args.ppo_epochs_per_iter,
                clip_range=args.ppo_clip,
                value_coef=args.ppo_value_coef,
                entropy_coef=args.ppo_entropy_coef,
                max_grad_norm=args.max_grad_norm,
            )

        stats = evaluate(model, eval_env, args.env_horizon)
        row = {
            "env_interactions": int(total_interactions),
            "bc_loss": train_loss,
            "ppo_loss": ppo_loss,
            **stats,
        }
        append_metrics_csv(metrics_path, row)
        eval_results.append(row)

        print(
            f"env_interactions={total_interactions} | "
            f"return={stats['mean_return']:.2f} | "
            f"success={stats['success_rate']:.3f} | "
            f"bc_loss={train_loss:.4f} | "
            f"ppo_loss={ppo_loss:.4f}"
        )
        if args.log_wandb:
            payload = {
                "env_interactions": total_interactions,
                "train/bc_loss": train_loss,
                "train/ppo_loss": ppo_loss,
            }
            payload.update({f"eval/{k}": v for k, v in stats.items()})
            wandb.log(payload)

        torch.save(model.state_dict(), os.path.join(run_dir, "latest_model.pt"))

    with open(os.path.join(run_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(eval_results, f)
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))

    if args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
