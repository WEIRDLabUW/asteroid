"""
AAWR (Asymmetric Advantage Weighted Regression) Training Algorithm.

Based on: "Real-world RL for Active Perception Behaviors" (Penn PAL Lab)
https://github.com/penn-pal-lab/aawr

The key idea is to use privileged information (goal) during critic training
to compute high-quality advantages, then use advantage-weighted BC for policy.

Algorithm:
1. Collect offline data with noisy expert (epsilon-greedy)
2. Train asymmetric critic (Q + V) using IQL with privileged goal info
3. Extract policy via AWR using advantage weights
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse

from experiments.config import parse_with_config
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import random
import tqdm
import gym
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt

from environments.create_envs import create_env
from datasets.dataset import collate_fn, SequenceDataset
from experiments.eval_policy import evaluate_policy_on_envs
from models import DecisionTransformer, AsymmetricCritic
from environments.rollout_policy import get_rollout_policy, NoisyExpertPolicy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Data Collection
# ============================================================================

def collect_aawr_data(envs, horizon, epsilon=0.25):
    """
    Collect offline data with noisy expert policy for AAWR.
    
    Args:
        envs: List of vectorized environments
        horizon: Number of steps to collect per environment
        epsilon: Probability of random action
    
    Returns:
        List of trajectory dicts with goal information
    """
    policy = NoisyExpertPolicy(epsilon=epsilon)
    trajs = []
    
    for env in tqdm.tqdm(envs, desc="Collecting AAWR data"):
        policy.set_env(env)
        state = env.reset()
        policy.reset()
        n_envs = env.num_envs
        
        states = []
        actions = []
        expert_actions = []
        rewards = []
        dones_list = []
        next_states = []

        for t in range(horizon):
            # Get noisy expert action
            action = policy.get_action(state)
            
            # Get true expert action for supervision
            if hasattr(env, "have_keys"):
                expert_action = env.opt_action(state, env.have_keys)
            else:
                expert_action = env.opt_action(state)

            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            expert_actions.append(expert_action)
            rewards.append(reward)
            dones_list.append(done)
            next_states.append(next_state)
            
            # Update policy context
            policy.update_context(state, action, reward, done)
            
            # Handle episode resets
            if np.any(done):
                next_state = env.reset()
            
            state = next_state

        # Stack arrays
        states = np.stack(states, axis=1)
        actions = np.stack(actions, axis=1)
        expert_actions = np.stack(expert_actions, axis=1)
        rewards = np.stack(rewards, axis=1)
        dones = np.stack(dones_list, axis=1)
        next_states = np.stack(next_states, axis=1)
        
        # Create trajectory dicts with goal info
        for k in range(n_envs):
            traj = {
                "states": states[k],
                "actions": actions[k],
                "expert_actions": expert_actions[k],
                "rewards": rewards[k],
                "dones": dones[k],
                "next_states": next_states[k],
                "goal": env._envs[k].goal.copy(),  # Privileged info
            }
            trajs.append(traj)
    
    return trajs


class AAWRDataset(torch.utils.data.Dataset):
    """Dataset for AAWR that includes goal information."""
    
    def __init__(self, trajs, config):
        self.trajs = trajs
        self.config = config
        self.horizon = config['horizon']
    
    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, index):
        traj = self.trajs[index]
        return {
            'states': torch.from_numpy(traj['states']).float(),
            'actions': torch.from_numpy(traj['actions']).float(),
            'expert_actions': torch.from_numpy(traj['expert_actions']).float(),
            'rewards': torch.from_numpy(traj['rewards']).float(),
            'dones': torch.from_numpy(traj['dones']).float(),
            'next_states': torch.from_numpy(traj['next_states']).float(),
            'goal': torch.from_numpy(traj['goal']).float(),
        }


def aawr_collate_fn(batch):
    """Collate function for AAWR dataset."""
    from torch.nn.utils.rnn import pad_sequence
    
    padded_batch = {}
    for key in batch[0]:
        if key == 'goal':
            # Goals don't need padding, just stack
            padded_batch[key] = torch.stack([item[key] for item in batch])
        else:
            padded_batch[key] = pad_sequence(
                [item[key] for item in batch], 
                batch_first=True
            )
    
    # Create attention mask
    lengths = torch.tensor([item['states'].shape[0] for item in batch])
    max_len = lengths.max()
    attention_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    
    padded_batch['attention_mask'] = attention_mask
    return padded_batch


# ============================================================================
# IQL Training (Critic)
# ============================================================================

def expectile_loss(pred, target, expectile=0.9):
    """
    Expectile regression loss for IQL.
    
    When expectile > 0.5, this emphasizes the upper quantile of the target
    distribution, effectively learning an optimistic value estimate.
    
    Args:
        pred: Predicted values
        target: Target values
        expectile: Expectile parameter (0.9 emphasizes 90th percentile)
    
    Returns:
        Loss value
    """
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff ** 2)).mean()


def train_iql_critic(
    critic,
    train_loader,
    args,
    save_dir,
):
    """
    Train the asymmetric critic using IQL.
    
    IQL training:
    - Q-network: TD loss with target Q
    - V-network: Expectile regression on Q values
    
    Args:
        critic: AsymmetricCritic model
        train_loader: DataLoader with AAWR dataset
        args: Training arguments
        save_dir: Directory to save checkpoints
    
    Returns:
        Trained critic
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize target network
    critic.init_target()
    
    # Optimizer
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    
    # Training loop
    global_step = 0
    
    for epoch in tqdm.tqdm(range(args.critic_epochs), desc="Training IQL Critic"):
        critic.train()
        epoch_stats = defaultdict(list)
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Flatten batch for transition-level training
            # batch['states']: (B, T, state_dim)
            # batch['actions']: (B, T, action_dim)
            # batch['rewards']: (B, T)
            # batch['dones']: (B, T)
            # batch['next_states']: (B, T, state_dim)
            # batch['goal']: (B, goal_dim)
            
            B, T = batch['states'].shape[:2]
            mask = batch['attention_mask']  # (B, T)
            
            # Expand goal to match sequence length
            goal_expanded = batch['goal'].unsqueeze(1).expand(-1, T, -1)  # (B, T, goal_dim)
            
            # Flatten all tensors
            states_flat = batch['states'].reshape(-1, batch['states'].shape[-1])
            actions_flat = batch['actions'].reshape(-1, batch['actions'].shape[-1])
            next_states_flat = batch['next_states'].reshape(-1, batch['next_states'].shape[-1])
            rewards_flat = batch['rewards'].reshape(-1)
            dones_flat = batch['dones'].reshape(-1)
            goal_flat = goal_expanded.reshape(-1, goal_expanded.shape[-1])
            mask_flat = mask.reshape(-1)
            
            # Only train on valid (non-padded) transitions
            valid_idx = mask_flat > 0
            if valid_idx.sum() == 0:
                continue
                
            states = states_flat[valid_idx]
            actions = actions_flat[valid_idx]
            next_states = next_states_flat[valid_idx]
            rewards = rewards_flat[valid_idx]
            dones = dones_flat[valid_idx]
            goals = goal_flat[valid_idx]
            
            # ==================== V-network update ====================
            # V-network is trained with expectile regression on Q values
            with torch.no_grad():
                # Get Q value for current (s, a) pair
                q_values = critic.q_value(states, actions, goals).squeeze(-1)
            
            v_values = critic.v_value(states, goals).squeeze(-1)
            v_loss = expectile_loss(v_values, q_values, args.expectile)
            
            # ==================== Q-network update ====================
            # Q-network: TD loss with bootstrapped target
            with torch.no_grad():
                # V(s') for bootstrapping
                next_v = critic.v_value(next_states, goals).squeeze(-1)
                # TD target: r + gamma * V(s') * (1 - done)
                q_target = rewards + args.gamma * next_v * (1 - dones)
            
            q_pred = critic.q_value(states, actions, goals).squeeze(-1)
            q_loss = F.mse_loss(q_pred, q_target)
            
            # Total loss
            loss = v_loss + q_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            
            # Update target network
            critic.update_target(tau=args.target_tau)
            
            # Log stats
            epoch_stats['v_loss'].append(v_loss.item())
            epoch_stats['q_loss'].append(q_loss.item())
            epoch_stats['total_loss'].append(loss.item())
            epoch_stats['q_mean'].append(q_pred.mean().item())
            epoch_stats['v_mean'].append(v_values.mean().item())
            
            global_step += 1
        
        # Log epoch stats
        if args.log_wandb:
            for k, v in epoch_stats.items():
                wandb.log({f"iql/{k}": np.mean(v), "iql/epoch": epoch})
        
        if epoch % max(1, args.critic_epochs // 10) == 0:
            print(f"Epoch {epoch}: Q_loss={np.mean(epoch_stats['q_loss']):.4f}, "
                  f"V_loss={np.mean(epoch_stats['v_loss']):.4f}")
        
        # Save checkpoint
        if epoch % max(1, args.critic_epochs // 5) == 0:
            torch.save(critic.state_dict(), os.path.join(save_dir, f"critic_epoch_{epoch}.pth"))
    
    # Save final critic
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
    
    return critic


# ============================================================================
# AWR Policy Training
# ============================================================================

def get_loss_mask(attention_mask, horizon):
    """Get mask for loss computation on last horizon tokens."""
    loss_mask = torch.zeros_like(attention_mask)
    for i in range(loss_mask.size(0)):
        non_zero_indices = torch.nonzero(attention_mask[i], as_tuple=False).squeeze()
        if len(non_zero_indices.shape) == 0:
            non_zero_indices = non_zero_indices.unsqueeze(0)
        if len(non_zero_indices) >= horizon:
            loss_mask[i, non_zero_indices[-horizon:]] = 1
        else:
            loss_mask[i, non_zero_indices] = 1
    return loss_mask


def train_awr_policy(
    model,
    critic,
    train_loader,
    args,
    save_dir,
    action_dim,
    env_horizon,
):
    """
    Train policy using Advantage Weighted Regression (AWR).
    
    AWR: weight = exp(A / temperature), where A = Q(s,a,g) - V(s,g)
    Policy loss: weighted cross-entropy with expert actions
    
    Args:
        model: Policy model (DecisionTransformer)
        critic: Trained AsymmetricCritic (frozen)
        train_loader: DataLoader with AAWR dataset
        args: Training arguments
        save_dir: Directory to save checkpoints
        action_dim: Action dimension
        env_horizon: Environment horizon
    
    Returns:
        Trained policy
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Freeze critic
    critic.eval()
    for param in critic.parameters():
        param.requires_grad = False
    
    # Setup optimizer with warmup
    total_steps = len(train_loader) * args.policy_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )
    
    eval_freq = max(1, int(args.eval_interval * args.policy_epochs))
    save_freq = max(1, int(args.save_interval * args.policy_epochs))
    
    for epoch in tqdm.tqdm(range(args.policy_epochs), desc="Training AWR Policy"):
        model.train()
        epoch_stats = defaultdict(list)
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get policy predictions
            pred_actions, _ = model(batch, sample_time=True)
            true_actions = batch['expert_actions']
            
            # Compute advantages using frozen critic
            B, T = batch['states'].shape[:2]
            goal_expanded = batch['goal'].unsqueeze(1).expand(-1, T, -1)
            
            with torch.no_grad():
                # Flatten for critic
                states_flat = batch['states'].reshape(-1, batch['states'].shape[-1])
                actions_flat = batch['actions'].reshape(-1, batch['actions'].shape[-1])
                goal_flat = goal_expanded.reshape(-1, goal_expanded.shape[-1])
                
                # Compute advantages
                advantages = critic.advantage(states_flat, actions_flat, goal_flat)
                advantages = advantages.reshape(B, T)  # (B, T)
                
                # Compute AWR weights: w = exp(A / temperature)
                # Clip advantages for numerical stability
                advantages_clipped = torch.clamp(advantages / args.awr_temperature, -10, 10)
                weights = torch.exp(advantages_clipped)
                
                # Optional: use indicator filter (only positive advantages)
                if args.awr_filter == "indicator":
                    weights = (advantages > 0).float()
                elif args.awr_filter == "exp_clamp":
                    weights = torch.clamp(weights, 0, 100)
            
            # Cross entropy loss for discrete actions
            action_loss = F.cross_entropy(
                pred_actions.reshape(-1, action_dim),
                true_actions.reshape(-1, action_dim),
                reduction='none'
            )
            action_loss = action_loss.reshape(B, T)
            
            # Apply loss mask (only last env_horizon tokens)
            loss_mask = get_loss_mask(batch['attention_mask'], env_horizon)
            
            # Weighted loss
            weighted_loss = action_loss * weights * loss_mask
            loss = weighted_loss.sum() / loss_mask.sum()
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Log stats
            epoch_stats['loss'].append(loss.item())
            epoch_stats['advantage_mean'].append(advantages.mean().item())
            epoch_stats['advantage_std'].append(advantages.std().item())
            epoch_stats['weight_mean'].append(weights.mean().item())
        
        # Log epoch stats
        if args.log_wandb:
            for k, v in epoch_stats.items():
                wandb.log({f"awr/{k}": np.mean(v), "awr/epoch": epoch})
            wandb.log({"awr/lr": optimizer.param_groups[0]['lr']})
        
        if epoch % eval_freq == 0:
            print(f"Epoch {epoch}: loss={np.mean(epoch_stats['loss']):.4f}, "
                  f"adv_mean={np.mean(epoch_stats['advantage_mean']):.4f}")
        
        # Save checkpoint
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAWR Training")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="aawr")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)
    
    # Data collection
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--n_envs", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=1000, help="Model/context horizon")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Noisy expert epsilon")
    parser.add_argument("--n_meta_episodes", type=int, default=10, help="Meta-episodes for data collection")
    
    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=40, help="Episodes for evaluation")
    
    # Model
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Critic (IQL)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--critic_epochs", type=int, default=100)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--expectile", type=float, default=0.9, help="IQL expectile")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target_tau", type=float, default=0.005, help="Target network update rate")
    
    # Policy (AWR)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--policy_epochs", type=int, default=100)
    parser.add_argument("--awr_temperature", type=float, default=3.0, help="AWR temperature")
    parser.add_argument("--awr_filter", type=str, default="exp_clamp", 
                        choices=["none", "indicator", "exp_clamp"])
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--eval_interval", type=float, default=0.1)
    parser.add_argument("--save_interval", type=float, default=0.1)
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="asteroid")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="results/aawr")

    args = parse_with_config(parser)

    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using device: {device}")
    
    # Save directory
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Create environments
    print(f"Creating environments: {args.env_name}")
    train_envs, test_envs, eval_envs = create_env(args.env_name, args.dataset_size, args.n_envs)
    
    state_dim = train_envs[0]._envs[0].state_dim
    action_dim = train_envs[0]._envs[0].action_dim
    env_horizon = train_envs[0]._envs[0].horizon
    goal_dim = len(train_envs[0]._envs[0].goal)
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Env horizon: {env_horizon}")
    print(f"Goal dim: {goal_dim}")
    print(f"Model horizon: {args.horizon}")

    # ========================================================================
    # Phase 1: Collect offline data with noisy expert
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 1: Collecting data with noisy expert")
    print("="*60)
    
    data_dir = os.path.join(save_dir, "data")
    train_data_path = os.path.join(data_dir, "train_data.pkl")
    test_data_path = os.path.join(data_dir, "test_data.pkl")
    
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print(f"Loading existing data from {data_dir}")
        with open(train_data_path, "rb") as f:
            train_trajs = pickle.load(f)
        with open(test_data_path, "rb") as f:
            test_trajs = pickle.load(f)
    else:
        os.makedirs(data_dir, exist_ok=True)
        
        # Collect data over n_meta_episodes per env
        data_horizon = args.n_meta_episodes * env_horizon
        
        train_trajs = collect_aawr_data(train_envs, data_horizon, args.epsilon)
        test_trajs = collect_aawr_data(test_envs, data_horizon, args.epsilon)
        
        with open(train_data_path, "wb") as f:
            pickle.dump(train_trajs, f)
        with open(test_data_path, "wb") as f:
            pickle.dump(test_trajs, f)
    
    print(f"Train trajectories: {len(train_trajs)}")
    print(f"Test trajectories: {len(test_trajs)}")

    # Create datasets
    config = {
        "horizon": args.horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    train_dataset = AAWRDataset(train_trajs, config)
    test_dataset = AAWRDataset(test_trajs, config)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=aawr_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=aawr_collate_fn,
    )

    # ========================================================================
    # Phase 2: Train IQL critic with privileged goal info
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 2: Training IQL Critic (asymmetric with goal)")
    print("="*60)
    
    critic = AsymmetricCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        hidden_dim=args.critic_hidden_dim,
    ).to(device)
    
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    critic = train_iql_critic(
        critic=critic,
        train_loader=train_loader,
        args=args,
        save_dir=os.path.join(save_dir, "critic"),
    )

    # ========================================================================
    # Phase 3: Train policy via AWR
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 3: Training Policy via AWR")
    print("="*60)
    
    # Model configuration
    continuous_action = isinstance(train_envs[0]._envs[0].action_space, gym.spaces.Box)
    model_args = {
        "horizon": 4000,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_embd": 128,
        "dropout": args.dropout,
        "shuffle": True,
        "test": False,
        "continuous_action": continuous_action,
        "gmm_heads": 1,
    }
    
    with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
        pickle.dump(model_args, f)
    
    model = DecisionTransformer(model_args).to(device)
    print(f"Policy parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_awr_policy(
        model=model,
        critic=critic,
        train_loader=train_loader,
        args=args,
        save_dir=os.path.join(save_dir, "policy"),
        action_dim=action_dim,
        env_horizon=env_horizon,
    )

    # ========================================================================
    # Phase 4: Evaluation
    # ========================================================================
    print("\n" + "="*60)
    print("Phase 4: Evaluation")
    print("="*60)
    
    eval_policy = get_rollout_policy(
        "decision_transformer",
        model=model,
        context_horizon=4000,
        env_horizon=env_horizon,
        context_accumulation=False,
        sliding_window=True,
    )
    
    eval_save_dir = os.path.join(save_dir, "eval")
    eval_horizon = args.eval_episodes * env_horizon
    
    eval_results = evaluate_policy_on_envs(
        eval_envs=eval_envs,
        policy=eval_policy,
        eval_horizon=eval_horizon,
        env_horizon=env_horizon,
        save_dir=eval_save_dir,
        env_name=args.env_name,
        plot=True,
    )
    
    # Log to wandb
    if args.log_wandb:
        final_mean = eval_results['mean_returns'][-1]
        final_std = eval_results['std_returns'][-1]
        wandb.log({
            "eval/final_return": final_mean,
            "eval/final_return_std": final_std,
            "eval/mean_return": np.mean(eval_results['mean_returns']),
        })
        
        # Log returns plot
        fig, ax = plt.subplots(figsize=(10, 6))
        episodes = np.arange(len(eval_results['mean_returns']))
        ax.plot(episodes, eval_results['mean_returns'], label='Mean Return', linewidth=2)
        ax.fill_between(
            episodes,
            eval_results['mean_returns'] - eval_results['std_returns'],
            eval_results['mean_returns'] + eval_results['std_returns'],
            alpha=0.2,
        )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(f'AAWR Eval Returns - {args.env_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"eval/returns_plot": wandb.Image(fig)})
        plt.close(fig)
    
    print(f"\nEvaluation complete - Final return: {eval_results['mean_returns'][-1]:.2f} "
          f"± {eval_results['std_returns'][-1]:.2f}")
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    if args.log_wandb:
        wandb.finish()

