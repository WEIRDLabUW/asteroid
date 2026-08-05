"""
Policy Evaluation Script.

Evaluates a trained policy on test environments and computes episode returns.
Can evaluate expert, random, or learned policies.
"""

import argparse
import os
import pathlib
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.create_envs import create_env
from datasets.collect_data import get_dagger_data, save_dagger_data
from environments.rollout_policy import get_rollout_policy
from models import DecisionTransformer


def load_model(checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on
    
    Returns:
        model: Loaded model
        model_args: Model configuration dictionary
    """
    # Try loading model_args from checkpoint dir or parent dir
    model_args_path = os.path.join(checkpoint_path, 'model_args.pkl')
    if not os.path.exists(model_args_path):
        model_args_path = os.path.join(checkpoint_path, '../model_args.pkl')
    
    with open(model_args_path, 'rb') as f:
        model_args = pickle.load(f)
    
    # Create and load model
    model = DecisionTransformer(model_args).to(device)
    
    # Find latest checkpoint
    checkpoints = list(pathlib.Path(checkpoint_path).glob('model_epoch_*.pth'))
    if not checkpoints:
        # Try best_model.pth
        best_model_path = os.path.join(checkpoint_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = best_model_path
        else:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
    else:
        checkpoint = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    
    return model, model_args


def compute_episode_returns(rewards, env_horizon):
    """
    Compute per-episode returns from trajectory rewards.
    
    Args:
        rewards: (B, T) array of rewards
        env_horizon: Number of steps per episode
    
    Returns:
        episode_returns: (B, num_episodes) array of returns
        mean_returns: (num_episodes,) mean return per episode
        std_returns: (num_episodes,) standard error per episode
    """
    episode_returns = []
    for reward_seq in rewards:
        done_indices = np.arange(0, len(reward_seq) + 1, env_horizon)
        seq_returns = [
            np.sum(reward_seq[done_indices[i]:done_indices[i + 1]]) 
            for i in range(len(done_indices) - 1)
        ]
        episode_returns.append(np.array(seq_returns))
    
    episode_returns = np.stack(episode_returns)  # (B, num_episodes)
    mean_returns = np.mean(episode_returns, axis=0)
    std_returns = np.std(episode_returns, axis=0) / np.sqrt(episode_returns.shape[0])
    
    return episode_returns, mean_returns, std_returns


def plot_returns(mean_returns, std_returns, env_name, save_path):
    """
    Plot episode returns with confidence bands.
    
    Args:
        mean_returns: (num_episodes,) mean return per episode
        std_returns: (num_episodes,) standard error per episode
        env_name: Environment name for title
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    episodes = np.arange(len(mean_returns))
    plt.plot(episodes, mean_returns, label='Mean Return', linewidth=2)
    plt.fill_between(
        episodes, 
        mean_returns - std_returns, 
        mean_returns + std_returns, 
        alpha=0.2, 
        label='Standard Error'
    )
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'Evaluation Returns on {env_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_policy_on_envs(
    eval_envs,
    policy,
    eval_horizon,
    env_horizon,
    save_dir,
    env_name,
    plot=True,
):
    """
    Evaluate a policy and save results.
    
    Args:
        eval_envs: List of evaluation environments
        policy: Policy to evaluate
        eval_horizon: Total evaluation horizon
        env_horizon: Steps per episode
        save_dir: Directory to save results
        env_name: Environment name
        plot: Whether to generate plots
    
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect trajectories (returns list of trajectory dicts)
    eval_trajs = get_dagger_data(eval_envs, policy, eval_horizon)
    save_dagger_data(eval_trajs, os.path.join(save_dir, 'eval_trajs.pkl'))
    
    # Extract rewards from trajectory list and stack into (B, T) array
    rewards = np.stack([traj['rewards'] for traj in eval_trajs])
    
    # Compute returns
    episode_returns, mean_returns, std_returns = compute_episode_returns(
        rewards, env_horizon
    )
    
    # Save returns
    np.savez(
        os.path.join(save_dir, 'eval_returns.npz'),
        episode_returns=episode_returns,
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    print(f"Saved evaluation returns to {os.path.join(save_dir, 'eval_returns.npz')}")
    
    # Plot if requested
    if plot:
        plot_path = os.path.join(save_dir, 'eval_returns.png')
        plot_returns(mean_returns, std_returns, env_name, plot_path)
        print(f"Saved evaluation plot to {plot_path}")
    
    return {
        'episode_returns': episode_returns,
        'mean_returns': mean_returns,
        'std_returns': std_returns,
        'trajs': eval_trajs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument('--exp_name', type=str, default='eval_policy')
    parser.add_argument('--env_name', type=str, default='darkroom-easy')
    parser.add_argument('--n_envs', type=int, default=1000)
    parser.add_argument('--eval_episodes', type=int, default=40)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--policy_type', type=str, choices=['', 'random', 'expert', 'decision_transformer'], default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--plot_returns', action='store_true')
    parser.add_argument('--temp', type=float, default=1.0)
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environments
    train_envs, test_envs, eval_envs = create_env(args.env_name, 10000, args.n_envs)
    assert len(eval_envs) == 1, f"Expected only one eval env, got {len(eval_envs)}"
    
    env_horizon = eval_envs[0]._envs[0].horizon
    eval_horizon = args.eval_episodes * env_horizon

    # Setup save directory
    save_dir = os.path.join(args.save_path, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model or use baseline policy
    if args.policy_type in ['expert', 'random']:
        model = None
        context_horizon = env_horizon
    else:
        model, model_args = load_model(args.checkpoint_path, device)
        context_horizon = model_args['horizon']
        args.policy_type = args.policy_type or 'decision_transformer'

    # Create policy
    policy = get_rollout_policy(
        args.policy_type, 
        model=model, 
        temp=args.temp, 
        context_horizon=context_horizon,
        env_horizon=env_horizon,
    )

    # Evaluate
    results = evaluate_policy_on_envs(
        eval_envs=eval_envs,
        policy=policy,
        eval_horizon=eval_horizon,
        env_horizon=env_horizon,
        save_dir=save_dir,
        env_name=args.env_name,
        plot=args.plot_returns,
    )
    
    print(f"\nEvaluation complete!")
    print(f"Mean return per episode: {results['mean_returns']}")
    print(f"Final episode return: {results['mean_returns'][-1]:.2f} ± {results['std_returns'][-1]:.2f}")
