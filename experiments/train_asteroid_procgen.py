"""ASTEROID on the Procgen maze benchmark (image / partial-grid observations).

Same on-policy asymmetric distillation as ``train_asteroid.py`` but with a CNN
Decision Transformer over procgen observations. Each iteration collects on-policy
student contexts, queries the privileged expert for labels, trains by conditional
imitation, and logs evaluation metrics and rollout videos.

Requires the optional ``procgen`` extra: ``uv sync --extra procgen``.
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb

from PIL import Image

from environments.rollout_policy import TransformerCNNPolicy
from models import DecisionTransformerCnn
from environments.procgen_env import make_maze_envs, _render_grid_obs


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy_on_envs_procgen(eval_envs, policy, eval_horizon,
                                    save_dir, dagger_step, eval_name,
                                    plot=True, panel_size=256):
    """
    Evaluate *policy* on the VecProcgenMaze *eval_envs*.

    * Runs one batch of parallel episodes up to *eval_horizon* steps.
    * Records per-env videos with two side-by-side panels:
      partial obs (rendered grid) | full RGB frame.
    * Returns (mean_return, std_return) across environments.
    """
    import imageio_ffmpeg

    os.makedirs(save_dir, exist_ok=True)

    obs, infos = eval_envs.reset()
    resets = np.ones(eval_envs.n, dtype=bool)
    policy.reset(resets)

    n = eval_envs.n
    done_flag = np.zeros(n, dtype=bool)
    episode_rewards = np.zeros(n, dtype=np.float32)
    successes = np.zeros(n, dtype=bool)
    episode_lengths = np.zeros(n, dtype=int)
    # each frame is (partial_rgb, full_rgb) – both uint8
    episode_frames = [[] for _ in range(n)]

    ps = panel_size
    pbar = tqdm.tqdm(total=n, desc=f"Evaluating {eval_name}", unit="step")
    for t in range(eval_horizon):
        for i in range(n):
            if not done_flag[i]:
                # rendered partial obs
                partial = _render_grid_obs(obs[i])  # uint8
                partial = np.array(Image.fromarray(partial).resize(
                    (ps, ps), Image.NEAREST))
                # full RGB from procgen
                rgb = infos[i]["full_obs"]
                rgb = _render_grid_obs(rgb) 
                # if rgb is None:
                #     rgb = _render_grid_obs(obs[i])
                rgb = np.array(Image.fromarray(rgb).resize(
                    (ps, ps), Image.NEAREST))
                episode_frames[i].append((partial, rgb))

        # prev_obs = obs
        actions = policy.get_action(obs)
        next_obs, rewards, dones, infos = eval_envs.step(actions)
        policy.update_context(obs, actions, rewards, dones)
        policy.reset(dones)
        obs = next_obs
        episode_lengths[~done_flag] += 1

        for i in range(n):
            if not done_flag[i]:
                episode_rewards[i] += rewards[i]
            if dones[i] and not done_flag[i]:
                done_flag[i] = True
                pbar.update(1)
                if rewards[i] > 0:
                    successes[i] = True

        if done_flag.all():
            break

    # save mp4 videos for up to 5 envs
    for i in range(min(n, 5)):
        if len(episode_frames[i]) == 0:
            continue
        vid_path = os.path.join(save_dir, f"episode_{i}.mp4")
        # build side-by-side frames
        vid_frames = []
        for partial, rgb in episode_frames[i]:
            vid_frames.append(np.concatenate([partial, rgb], axis=1))
        h, w = vid_frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(
            vid_path, (w, h), fps=10, pix_fmt_in="rgb24")
        writer.send(None)
        for frame in vid_frames:
            writer.send(frame.tobytes())
        writer.close()

        if wandb.run is not None:
            wandb.log({
                f"step{dagger_step}_eval/video_{eval_name}_episode_{i}_video_{'success' if successes[i] else 'failure'}":
                    wandb.Video(vid_path, format="mp4")
            })

    mean_ret = float(np.mean(episode_rewards))
    std_ret = float(np.std(episode_rewards))
    success_rate = float(np.mean(successes))
    mean_length = float(np.mean(episode_lengths))
    mean_success_length = float(np.mean(episode_lengths[successes])) if successes.any() else 0.0
    stats = {
        "mean_return": mean_ret,
        "std_return": std_ret,
        "success_rate": success_rate,
        "mean_length": mean_length,
        "mean_success_length": mean_success_length
    }
    return stats

# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(torch.utils.data.Dataset):
    """Dataset of variable-length trajectories collected from procgen maze."""

    def __init__(self, trajectories, sample_steps=False):
        self.trajectories = trajectories
        self.total_steps = sum(len(traj['actions']) for traj in trajectories)
        self.sample_steps = False
        self.step_to_traj = []
        if self.sample_steps:
            for traj_idx, traj in enumerate(trajectories):
                for step_idx in range(len(traj['actions'])):
                    self.step_to_traj.append((traj_idx, step_idx))

    def __len__(self):
        return len(self.trajectories) if not self.sample_steps else self.total_steps

    def __getitem__(self, idx):
        if self.sample_steps:
            traj_idx, step_idx = self.step_to_traj[idx]
            traj = self.trajectories[traj_idx]
            steps = np.array([step_idx])
        else:
            traj = self.trajectories[idx]
            steps = np.arange(len(traj['actions'])) # (T,)
        return {
            # observations: (T, H, W, C) float32
            "observations": torch.tensor(
                np.array(traj['observations'])[steps], dtype=torch.float32),
            # actions: (T,) long  – discrete action ids 0..3
            "actions": torch.tensor(
                np.array(traj['actions'])[steps], dtype=torch.long),
            # rewards: (T,)
            "rewards": torch.tensor(
                np.array(traj['rewards'])[steps], dtype=torch.float32),
            # dones: (T,)
            "dones": torch.tensor(
                np.array(traj['dones'])[steps], dtype=torch.float32),
            # expert_actions: (T,) long
            # "expert_actions": torch.tensor(
            #     np.array(traj['expert_actions']), dtype=torch.long),
            # expert_mask: (T,) float – 1 where expert was used
            "expert_mask": torch.tensor(
                np.array(traj['expert_mask'])[steps], dtype=torch.float32),
        }

# class TrajectoryDataset(torch.utils.data.Dataset):
#     """Dataset of variable-length trajectories collected from procgen maze."""

#     def __init__(self, trajectories, sample_steps=False):
#         self.trajectories = trajectories
#         expert_masks = [traj['expert_mask'] for traj in trajectories]
#         self.supervision_steps = sum(np.sum(mask) for mask in expert_masks)
#         self.idx_to_traj_step = []
#         for traj_idx, traj in enumerate(trajectories):
#             for step_idx in range(len(traj['actions'])):
#                 if traj['expert_mask'][step_idx]:  # only include steps with expert supervision
#                     self.idx_to_traj_step.append((traj_idx, step_idx))
#         assert len(self.idx_to_traj_step) == self.supervision_steps, "Mismatch in supervision steps count"


#     def __len__(self):
#         # return len(self.trajectories) if not self.sample_steps else self.total_steps
#         return self.supervision_steps

#     def __getitem__(self, idx):
#         traj_idx, step_idx = self.idx_to_traj_step[idx]
#         traj = self.trajectories[traj_idx]
#         observations = np.array(traj['observations'])[:step_idx+1]
#         actions = np.array(traj['actions'])[:step_idx+1]
#         rewards = np.array(traj['rewards'])[:step_idx+1]
#         dones = np.array(traj['dones'])[:step_idx+1]
#         expert_mask = np.zeros(step_idx+1, dtype=np.float32)
#         expert_mask[step_idx] = 1.0
#         # breakpoint()
#         return {
#             # observations: (T, H, W, C) float32
#             "observations": torch.tensor(observations, dtype=torch.float32),
#             # actions: (T,) long  – discrete action ids 0..3
#             "actions": torch.tensor(actions, dtype=torch.long),
#             # rewards: (T,)
#             "rewards": torch.tensor(rewards, dtype=torch.float32),
#             # dones: (T,)
#             "dones": torch.tensor(dones, dtype=torch.float32),
#             "expert_mask": torch.tensor(
#                 expert_mask, dtype=torch.float32),
#         }


def split_trajectories(trajectories, train_ratio=0.8, seed=42):
    """Split trajectories into train/val lists with deterministic shuffling."""
    n = len(trajectories)
    if n == 0:
        return [], []
    if n == 1:
        return trajectories, []

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_train = max(1, min(n_train, n - 1))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]
    return train_trajs, val_trajs

class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples
    copied from https://github.com/pytorch/pytorch/issues/2576
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        weights_np = self.weights.numpy()
        weights_sum = torch.sum(self.weights).numpy()
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=weights_np / weights_sum,
            replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())
    
class MultiTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, sampling_ratios):
        super().__init__()
        self.datasets = datasets        
        self.dataset_lengths = [len(d) for d in datasets]
        self.sampling_ratios = sampling_ratios
        normalized_ratios = np.array(sampling_ratios) / np.sum(sampling_ratios)
        weights = []
        for length, ratio in zip(self.dataset_lengths, normalized_ratios):
            weights.extend([ratio / length] * length)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        # self.weighted_sampler = CustomWeightedRandomSampler(self.weights, num_samples=len(self.weights), replacement=True)
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
    
    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, global_idx):
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], global_idx, side='right')
        local_idx = global_idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


def collate_fn_procgen(batch):
    """Pad variable-length trajectories and build attention / expert masks."""
    from torch.nn.utils.rnn import pad_sequence

    padded = {}
    for key in batch[0]:
        padded[key] = pad_sequence(
            [item[key] for item in batch], batch_first=True)

    lengths = torch.tensor([item['actions'].shape[0] for item in batch])
    max_len = int(lengths.max())
    attn = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, l in enumerate(lengths):
        attn[i, :l] = 1.0
    padded['attention_mask'] = attn
    return padded


# def get_procgen_dataset(env, n_trajs, eval_policy=None, expert_p=1.0,
#                         max_t=500, n_video_trajs=3):
def get_procgen_dataset(env, n_trajs, eval_policy, exploration_steps_range,
                        max_t=500, n_video_trajs=3):
    """Collect trajectories. Returns (list_of_traj_dicts, dataset)."""
    # if eval_policy is None:
    #     assert expert_p == 1.0
    # else:
    resets = np.ones(env.n, dtype=bool)
    eval_policy.reset(resets)

    def _new():
        return {'observations': [], 'actions': [], 'rewards': [],
                'dones': [], 'expert_mask': [],
                '_full_obs': [], '_rgb': [], '_opt_grid': []}

    all_trajs = []
    trajs = [_new() for _ in range(env.n)]
    obs, infos = env.reset()
    # is_expert = np.zeros(env.n, dtype=bool)
    # max_exploration_steps = 
    min_exploration_steps = exploration_steps_range[0]
    max_exploration_steps = exploration_steps_range[1]

    exploration_steps = np.random.randint(min_exploration_steps, max_exploration_steps + 1, size=env.n)
    current_exploration_steps = np.zeros(env.n, dtype=int)
    pbar = tqdm.tqdm(total=n_trajs, desc="Collecting trajectories", unit="traj")

    while len(all_trajs) < n_trajs:
        # pick actions
        acts = np.zeros(env.n, dtype=np.int32)
        # expert_acts = np.zeros(env.n, dtype=np.int32)
        # is_expert = is_expert | (np.random.random(env.n) < expert_p)
        current_exploration_steps += 1
        use_expert = current_exploration_steps >= exploration_steps
        if ~ use_expert.all():
            policy_action = eval_policy.get_action(obs)
        for i in range(env.n):
            if use_expert[i]:
                acts[i] = infos[i].get('opt_action', 0)
            else:
                acts[i] = policy_action[i]

        # prev_obs = obs
        next_obs, rews, dones, next_infos = env.step(acts)
        if eval_policy is not None:
            eval_policy.update_context(obs, acts, rews, dones)

        for i in range(env.n):
            save_vid = len(all_trajs) + i < n_video_trajs
            trajs[i]['observations'].append(obs[i].copy())
            trajs[i]['actions'].append(int(acts[i]))
            trajs[i]['rewards'].append(float(rews[i]))
            trajs[i]['dones'].append(bool(dones[i]))
            # trajs[i]['expert_actions'].append(int(expert_acts[i]))
            trajs[i]['expert_mask'].append(bool(use_expert[i]))
            trajs[i]['_full_obs'].append(
                infos[i].get('full_obs'))
            # trajs[i]['_rgb'].append(
            #     infos[i].get('rgb') if save_vid else None)
            trajs[i]['_opt_grid'].append(
                infos[i].get('opt_grid') if save_vid else None)

            if dones[i]:
                traj_return = sum(trajs[i]['rewards'])
                save_flag = traj_return > 0 and current_exploration_steps[i] >= min_exploration_steps and len(trajs[i]['actions']) > 2
                # make sure that expert actions are atleast 1% of the trajectory
                save_flag = save_flag and np.mean(trajs[i]['expert_mask']) >= 0.01
                if save_flag:  # only keep successful trajectories
                    all_trajs.append(trajs[i])
                    pbar.update(1)
                trajs[i] = _new()
                # is_expert[i] = False
                current_exploration_steps[i] = 0
                exploration_steps[i] = np.random.randint(min_exploration_steps, max_exploration_steps + 1)

        obs = next_obs
        infos = next_infos
        
        if eval_policy is not None:
            eval_policy.reset(dones)

        if all(len(t['observations']) > max_t for t in trajs):
            break

    # # flush partial
    # for i in range(env.n):
    #     if len(trajs[i]['actions']) > 0:
    #         all_trajs.append(trajs[i])

    all_trajs = all_trajs[:n_trajs]
    dataset = TrajectoryDataset(all_trajs, sample_steps=False)
    return all_trajs, dataset


ACT_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def save_dataset_videos(trajs, save_dir, dagger_step, visibility=None,
                        n_videos=5, panel_size=256):
    """Save annotated videos showing only the observation for each timestep.

    Each frame is the rendered binary obs (or raw RGB if no visibility),
    labelled with action, expert/policy source, and cumulative return.
    """
    import imageio_ffmpeg
    from PIL import ImageDraw

    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for traj in trajs:
        if count >= n_videos:
            break

        T = len(traj['observations'])
        if T == 0:
            continue

        combined = []
        cum_ret = 0.0
        for t in range(T):
            # obs_t = traj['observations'][t]
            obs_t = traj['_full_obs'][t]

            # render obs as RGB image
            # if visibility is not None:
            img = _render_grid_obs(obs_t)  # (wd*40, wd*40, 3) uint8
            # else:
            #     img = obs_t.astype(np.uint8) if obs_t.dtype != np.uint8 \
            #         else obs_t

            # resize to panel_size
            img = np.array(Image.fromarray(img).resize(
                (panel_size, panel_size), Image.NEAREST))

            # build label
            cum_ret += traj['rewards'][t]
            is_expert = traj['expert_mask'][t]
            act_id = traj['actions'][t]
            act_name = ACT_NAMES[act_id] if act_id < len(ACT_NAMES) \
                else str(act_id)
            src = "EXPERT" if is_expert else "POLICY"
            txt = f"t={t} {src} a={act_name} R={cum_ret:.1f}"

            # draw label
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            color = (0, 255, 0) if is_expert else (255, 100, 100)
            draw.text((4, 4), txt, fill=color)
            combined.append(np.array(pil_img))

        # write mp4
        h, w = combined[0].shape[:2]
        vid_path = os.path.join(save_dir, f"dataset_traj_{count}.mp4")
        writer = imageio_ffmpeg.write_frames(
            vid_path, (w, h), fps=6, pix_fmt_in="rgb24")
        writer.send(None)
        for frame in combined:
            writer.send(frame.tobytes())
        writer.close()
        # print(f"saved {vid_path} ({T} frames)")

        if wandb.run is not None:
            wandb.log({
                f"step{dagger_step}_dataset/video_traj_{count}_video":
                    wandb.Video(vid_path, format="mp4")
            })
        count += 1
    # print(f"Saved {count} dataset trajectory videos to {save_dir}")


def get_optimizer_scheduler(model, total_steps, lr, warmup_ratio):
    """Create optimizer with warmup + cosine decay schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = int(total_steps * warmup_ratio)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )
    
    return optimizer, scheduler


def train_step(
    step_id,
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loaders_by_step,
    save_dir,
    args,
    device,
    action_dim,
    env_horizon,
):
    """
    Train model for one DAgger step.
    
    Args:
        step_id: Current DAgger iteration
        model: Model to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        val_loaders_by_step: List[(source_step_id, val_loader)]
        save_dir: Directory to save checkpoints
        args: Training arguments
        device: Device to train on
        action_dim: Action dimension
        env_horizon: Environment horizon (steps per episode)
    
    Returns:
        Trained model (best or final depending on args)
    """
    step_save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
    os.makedirs(step_save_dir, exist_ok=True)
    
    eval_freq = max(1, int(args.eval_interval * args.num_epochs))
    save_freq = max(1, int(args.save_interval * args.num_epochs))
    
    def forward(batch):
        """Compute cross-entropy loss for discrete actions."""
        batch = {k: v.to(device) for k, v in batch.items()}
        # breakpoint()
        # Model expects dict with 'states', 'actions', 'rewards', 'dones'
        # states: (B, T, H, W, C), actions: one-hot (B, T, A)
        B, T = batch['observations'].shape[:2]
        model_input = {
            'states': batch['observations'],   # (B, T, H, W, C)
            'actions': batch['actions'],         # (B, T, A)
            'rewards': batch['rewards'],        # (B, T)
            'dones': batch['dones'],            # (B, T)
        }

        pred_logits = model(model_input)  # (B, T, A)

        # true_actions = batch['expert_actions']  # (B, T) long
        true_actions = batch['actions']  # (B, T) long

        # Mask: only count loss where attention_mask AND expert_mask are 1
        loss_mask = batch['attention_mask'] * batch['expert_mask']  # (B, T)
        # Per-token cross entropy
        action_loss = F.cross_entropy(
            pred_logits.reshape(-1, action_dim),
            true_actions.reshape(-1),
            reduction='none',
        )  # (B*T,)
        action_loss = action_loss.reshape(B, T)

        if loss_mask.sum() > 0:
            loss = (action_loss * loss_mask).sum() / loss_mask.sum()
        else:
            loss = action_loss.mean()

        return loss, {"loss": loss.item()}
    
    for epoch in tqdm.tqdm(range(args.num_epochs),
                           desc=f"Training DAgger Step {step_id}"):
        model.train()
        train_stats = defaultdict(list)
        grad_norms = []

        for batch in train_loader:
            loss, stats = forward(batch)

            optimizer.zero_grad()
            loss.backward()

            if args.log_wandb:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norms.append(total_norm ** 0.5)

            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            for k, v in stats.items():
                train_stats[k].append(v)

        if args.log_wandb:
            train_payload = {
                f"step{step_id}/train_{k}": float(np.mean(v))
                for k, v in train_stats.items()
            }
            train_payload[f"step{step_id}/train_lr"] = \
                float(optimizer.param_groups[0]['lr'])
            if len(grad_norms) > 0:
                train_payload[f"step{step_id}/train_grad_norm"] = \
                    float(np.mean(grad_norms))
            train_payload["dagger_step"] = step_id
            train_payload["epoch"] = epoch
            wandb.log(train_payload)

        if len(val_loaders_by_step) > 0 and \
                (epoch % eval_freq == 0 or epoch == args.num_epochs - 1):
            model.eval()
            val_losses = {}
            with torch.no_grad():
                for source_step, val_loader in val_loaders_by_step:
                    source_stats = defaultdict(list)
                    for batch in val_loader:
                        _, stats = forward(batch)
                        for k, v in stats.items():
                            source_stats[k].append(v)
                    if len(source_stats["loss"]) > 0:
                        val_losses[source_step] = \
                            float(np.mean(source_stats["loss"]))

            if args.log_wandb and len(val_losses) > 0:
                val_payload = {
                    f"step{step_id}/val_source_step_{src}_loss": v
                    for src, v in val_losses.items()
                }
                val_payload[f"step{step_id}/val_mean_loss"] = \
                    float(np.mean(list(val_losses.values())))
                val_payload["dagger_step"] = step_id
                val_payload["epoch"] = epoch
                wandb.log(val_payload)

            # if len(val_losses) > 0:
            #     loss_msg = ", ".join(
            #         [f"s{src}: {loss:.4f}"
            #          for src, loss in sorted(val_losses.items())]
            #     )
                # print(f"Epoch {epoch} - Val loss by source step: {loss_msg}")

        if epoch % save_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join(step_save_dir, f"model_epoch_{epoch}.pth"),
            )

    return model
# def data_step(save_dir, step_id, train_envs, test_envs, rollout_policy, horizon,
#                normalize_actions=False, action_stats=None):
#     """
#     Collect data for one DAgger step.
    
#     Args:
#         save_dir: Directory to save/load data
#         step_id: Current DAgger iteration
#         train_envs: Training environments
#         test_envs: Test environments
#         rollout_policy: Policy to use for data collection
#         horizon: Horizon for data collection
#         normalize_actions: Whether to normalize actions
#         action_stats: Optional action stats from first iteration (mean, std)
    
#     Returns:
#         train_dataset, test_dataset
#     """
#     step_save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
#     os.makedirs(step_save_dir, exist_ok=True)
    
#     # Check if data already exists
#     train_path = os.path.join(step_save_dir, "train_dataset.pkl")
#     test_path = os.path.join(step_save_dir, "test_dataset.pkl")
    
#     if os.path.exists(train_path) and os.path.exists(test_path):
#         print(f"Loading existing data from {step_save_dir}")
#         with open(train_path, "rb") as f:
#             train_dataset = pickle.load(f)
#         with open(test_path, "rb") as f:
#             test_dataset = pickle.load(f)
#         # Re-apply normalization with provided stats if needed
#         if normalize_actions and action_stats is not None:
#             train_dataset.action_stats = action_stats
#             train_dataset.action_mean = action_stats['mean']
#             train_dataset.action_std = action_stats['std']
#             train_dataset.normalize_actions = True
#             test_dataset.action_stats = action_stats
#             test_dataset.action_mean = action_stats['mean']
#             test_dataset.action_std = action_stats['std']
#             test_dataset.normalize_actions = True
#     else:
#         print(f"Collecting new data for step {step_id}")
#         train_dataset, test_dataset = get_dagger_dataset(
#             train_envs, test_envs, rollout_policy, horizon,
#             normalize_actions=normalize_actions,
#             action_stats=action_stats
#         )
#         with open(train_path, "wb") as f:
#             pickle.dump(train_dataset, f)
#         with open(test_path, "wb") as f:
#             pickle.dump(test_dataset, f)

#     return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Context Accumulation Training for Procgen Maze")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="asteroid_procgen")
    parser.add_argument("--env_name", type=str, default="maze")
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--dagger_steps", type=int, default=100)
    parser.add_argument("--n_train_envs", type=int, default=16)
    parser.add_argument("--n_eval_envs", type=int, default=100)
    parser.add_argument("--visibility", type=int, default=7,
                        help="Partial-obs window size (e.g. 7). "
                             "None/0 = full 64x64 RGB obs.")
    parser.add_argument("--train_start_level", type=int, default=0,
                        help="First procgen level seed for training envs.")
    parser.add_argument("--train_num_levels", type=int, default=1000,
                        help="Number of procgen levels available to training envs.")
    parser.add_argument("--eval_start_level", type=int, default=1000,
                        help="First procgen level seed for eval envs.")
    parser.add_argument("--eval_num_levels", type=int, default=1000,
                        help="Number of procgen levels available to eval envs.")

    # Model
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--eval_interval", type=float, default=0.1)
    parser.add_argument("--save_interval", type=float, default=0.1)

    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="asteroid-procgen")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Paths
    parser.add_argument("--save_dir", type=str, default="results/asteroid_procgen")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = os.path.join(
        args.save_dir,
        f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Create environments  (VecProcgenMaze from procgen_env.py)
    # ------------------------------------------------------------------
    vis = args.visibility if args.visibility else None
    train_env, eval_env = make_maze_envs(
        n_train=args.n_train_envs,
        n_eval=args.n_eval_envs,
        train_start=args.train_start_level,
        train_levels=args.train_num_levels,
        eval_start=args.eval_start_level,
        eval_levels=args.eval_num_levels,
        visibility=vis,
    )
    obs_shape = tuple(train_env.observation_space.shape)  # (H, W, C)
    action_dim = train_env.action_space.n  # 4
    # env_horizon = 500  # procgen maze max episode steps (easy mode)
    env_horizon = 600  # procgen maze max episode steps (easy mode)
    print(f"Obs shape: {obs_shape}, Action dim: {action_dim}, "
          f"Env horizon: {env_horizon}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_horizon = 2 * env_horizon
    model_args = {
        "horizon": model_horizon,
        "obs": obs_shape,          # (H, W, C)
        "action_dim": action_dim,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "shuffle": True,
        "test": False,
    }
    with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
        pickle.dump(model_args, f)

    model = DecisionTransformerCnn(model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Optimizer (single schedule across all dagger steps)
    # ------------------------------------------------------------------
    # total_steps = (args.dataset_size * args.num_epochs) // args.batch_size
    # optimizer, scheduler = get_optimizer_scheduler(model, total_steps, args.lr, args.warmup_ratio)

    # # ------------------------------------------------------------------
    # # Expert-p annealing
    # # ------------------------------------------------------------------
    # expert_p = 1.0
    # min_expert_p = 0.05
    # expert_p_decay = (expert_p - min_expert_p) / max(1, args.dagger_steps - 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    # eval_policy = None  # None means pure expert on first step
    training_datasets = []
    validation_datasets = []

    # exploration_steps_curriculum = [
    #     (0, 0),
    #     (5, 20),
    #     (20, 40),
    #     (40, 60),
    #     (60, 100)
    # ]
    exploration_steps_curriculum = [
        # (0, 0),
        # (10, 30),
        # (20, 60),
        # (40, 120),
        # (80, 200)
        (0, 0),
        (25, 50),
        (50, 100),
        (100, 200),
        (200, 350),
        (350, 500),
        (400, 600)
    ]
    # create a curriculum that starts with 0-0 exploration steps, and then proceeds based on exploration steps
    # curriculum_start = 0
    # curriculum_end = env_horizon
    # delta = int((curriculum_end // args.dagger_steps))
    # exploration_steps_curriculum = [(0,0)]
    # sampling_ratio_curriculum = [(1.0,)]
    # for step_idx in range(1, args.dagger_steps):
    #     exploration_steps_curriculum.append((curriculum_start, curriculum_start + delta))
    #     curriculum_start += delta
    #     sampling_ratio = np.arange(step_idx + 1)+1
    #     sampling_ratio = sampling_ratio / sampling_ratio.sum()
    #     sampling_ratio_curriculum.append(tuple(sampling_ratio))
    #     # delta = int(curriculum_end * (step_idx + 1) / args.dagger_steps) - curriculum_start
    # lr_curriculum = [1e-5] * args.dagger_steps
    # lr_curriculum[0] = 1e-4  # start with higher lr for first step
    # print(f"Exploration steps curriculum: {exploration_steps_curriculum}")
    # print(f"Sampling ratio curriculum: {sampling_ratio_curriculum}")
    # print(f"Learning rate curriculum: {lr_curriculum}")

    # sampling_ratio_curriculum = [
    #     (1.0, ),
    #     (0.25, 0.75),
    #     (0.2, 0.3, 0.5),
    #     (0.1, 0.2, 0.3, 0.4),
    #     (0.05, 0.15, 0.25, 0.35, 0.4)
    # ]
    sampling_ratio_curriculum = [
        (1.0, ),
        (0.5, 0.5),
        (0.33, 0.33, 0.34),
        (0.25, 0.25, 0.25, 0.25),
        (0.2, 0.2, 0.2, 0.2, 0.2)
    ]
    lr_curriculum = [
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
    ]
    for step_idx in range(args.dagger_steps):
        print(f"\n{'=' * 60}")
        # print(f"DAgger Step {step_idx}/{args.dagger_steps}  "
            #   f"(expert_p={expert_p:.3f})")
        # print step, exploration steps range, sampling ratio, lr
        print(f"DAgger Step {step_idx}/{args.dagger_steps}")
        print(f"Exploration steps range: {exploration_steps_curriculum[step_idx]}")
        print(f"Sampling ratio: {sampling_ratio_curriculum[step_idx]}")
        print(f"Learning rate: {lr_curriculum[step_idx]:.2e}")
        print(f"{'=' * 60}")

        # 1. Collect data
        data_collection_policy = TransformerCNNPolicy(
            model=model,
            context_horizon=model_horizon,
            temp=1.0
        )
        all_trajs, _ = get_procgen_dataset(
            train_env,
            n_trajs=args.dataset_size,
            eval_policy=data_collection_policy,
            # expert_p=expert_p,
            exploration_steps_range=exploration_steps_curriculum[step_idx]
        )
        current_train_trajs, current_val_trajs = split_trajectories(
            all_trajs, train_ratio=0.8, seed=args.seed + step_idx
        )
        current_train_dataset = TrajectoryDataset(
            current_train_trajs, sample_steps=False
        )
        current_val_dataset = TrajectoryDataset(
            current_val_trajs, sample_steps=False
        )
        training_datasets.append(current_train_dataset)
        validation_datasets.append(current_val_dataset)
        print(f"Collected {len(all_trajs)} trajectories")
        print(f"Split -> train: {len(current_train_dataset)}, "
              f"val: {len(current_val_dataset)}")

        sampling_ratio = sampling_ratio_curriculum[step_idx]
        print(f"Sampling ratio for training: {sampling_ratio}")

        if args.log_wandb:
            dataset_stats = {
                f"step{step_idx}_dataset/current_total_trajs":
                    len(all_trajs),
                f"step{step_idx}_dataset/current_train_trajs":
                    len(current_train_dataset),
                f"step{step_idx}_dataset/current_val_trajs":
                    len(current_val_dataset),
                f"step{step_idx}_dataset/combined_train_trajs":
                    int(sum(len(d) for d in training_datasets)),
                f"step{step_idx}_dataset/combined_val_trajs":
                    int(sum(len(d) for d in validation_datasets)),
                "dagger_step": step_idx,
            }
            for source_step, val_ds in enumerate(validation_datasets):
                dataset_stats[
                    f"step{step_idx}_dataset/val_source_step_{source_step}_trajs"
                ] = len(val_ds)
            wandb.log(dataset_stats)

        # 1b. Log a few annotated dataset trajectory videos
        data_vid_dir = os.path.join(
            save_dir, f"dagger_step_{step_idx}", "dataset_videos")
        save_dataset_videos(
            all_trajs, data_vid_dir, step_idx,
            visibility=vis, n_videos=5,
        )

        train_dataset = MultiTrajectoryDataset(training_datasets, sampling_ratio)
        weighted_sampler = CustomWeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset.weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_procgen,
            sampler=weighted_sampler
        )
        val_loaders_by_step = []
        for source_step, val_dataset in enumerate(validation_datasets):
            if len(val_dataset) == 0:
                continue
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn_procgen,
            )
            val_loaders_by_step.append((source_step, val_loader))

        # 2. Train
        model = DecisionTransformerCnn(model_args).to(device)
        total_steps = (len(train_dataset.weights) * args.num_epochs) // args.batch_size
        optimizer, scheduler = get_optimizer_scheduler(model, total_steps, lr_curriculum[step_idx], args.warmup_ratio)

        model = train_step(
            step_idx, model, optimizer, scheduler,
            train_loader, val_loaders_by_step,
            save_dir, args, device,
            action_dim, env_horizon,
        )

        # 3. Build evaluation policy from trained model
        print(f"\nEvaluating after DAgger step {step_idx}...")
        eval_policy = TransformerCNNPolicy(
            model=model,
            context_horizon=model_horizon,
            temp=1.0
        )
        # eval_policy_adapter = _PolicyAdapter(transformer_policy)

        # 4. Evaluate
        eval_save_dir = os.path.join(
            save_dir, f"dagger_step_{step_idx}", "eval")
        eval_stats = evaluate_policy_on_envs_procgen(
            eval_envs=eval_env,
            # eval_envs=train_env,  # evaluate on training envs to see improvement across steps
            policy=eval_policy,
            eval_horizon=env_horizon,
            save_dir=eval_save_dir,
            dagger_step=step_idx,
            eval_name="temp_1.0",
        )

        # print(f"Eval return: {mean_ret:.2f} ± {std_ret:.2f}")

        ## Low temp eval
        eval_policy_low_temp = TransformerCNNPolicy(
            model=model,
            context_horizon=model_horizon,
            temp=0.1
        )
        eval_low_temp_stats = evaluate_policy_on_envs_procgen(
            eval_envs=eval_env,
            policy=eval_policy_low_temp,
            eval_horizon=env_horizon,
            save_dir=os.path.join(eval_save_dir, "low_temp"),
            dagger_step=step_idx,
            eval_name="temp_0.1",
        )
        # print(f"Low-temp eval return: {mean_ret_low:.2f} ± {std_ret_low:.2f}")

        if args.log_wandb:
            eval_payload = {"dagger_step": step_idx}
            eval_payload.update({
                f"eval/{eval_name}_{metric}": value
                for eval_name, stats in [("temp_1.0", eval_stats),
                                         ("temp_0.1", eval_low_temp_stats)]
                for metric, value in stats.items()
            })
            # eval_payload = {
            #     "dagger_step": step_idx,
            #     "eval/temp_1.0_mean_return":
            #         mean_ret,
            #     "eval/temp_1.0_std_return":
            #         std_ret,
            #     "eval/temp_1.0_success_rate":
            #         success_rate,
            #     "eval/temp_0.1_mean_return":
            #         mean_ret_low,
            #     "eval/temp_0.1_std_return":
            #         std_ret_low,
            #     "eval/temp_0.1_success_rate":
            #         success_rate_low,
            #     # f"eval/curriculum_exploration_min":
            #     #     exploration_steps_curriculum[step_idx][0],
            #     # f"step{step_idx}_eval/curriculum_exploration_max":
            #     #     exploration_steps_curriculum[step_idx][1],
            #     # f"step{step_idx}_eval/curriculum_learning_rate":
            #     #     lr_curriculum[step_idx],
            # }
            for ratio_idx, ratio_val in enumerate(sampling_ratio):
                eval_payload[
                    f"eval/curriculum_sampling_ratio_{ratio_idx}"
                ] = ratio_val
            wandb.log(eval_payload)

        # 5. Decay expert probability
        # expert_p = max(min_expert_p, expert_p - expert_p_decay)
        # print(f"Expert probability for next step: {expert_p:.4f}")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print(f"\nTraining complete! Results saved to {save_dir}")

    if args.log_wandb:
        wandb.finish()
