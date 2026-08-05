"""
RL^2 Baseline using Recurrent PPO from sb3_contrib.

RL^2 learns to adapt within a meta-episode by conditioning on reward history.
The observation is augmented with (prev_reward, prev_done) to provide adaptation signal.
"""

import argparse

from experiments.config import parse_with_config
import os
import pickle
import random

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO

from environments.create_envs import create_env
from environments.meta_env import MetaEnv, MetaVecEnv


class WandbCallback(BaseCallback):
    """Callback for logging to wandb."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                wandb.log({
                    "train/mean_reward": np.mean(ep_rewards),
                    "train/std_reward": np.std(ep_rewards),
                    "train/timesteps": self.num_timesteps,
                })
        return True


def extract_env_pool(vec_envs):
    """Extract individual environments from list of VecEnvs."""
    env_pool = []
    for vec_env in vec_envs:
        # VecEnvs have ._envs attribute containing individual envs
        if hasattr(vec_env, '_envs'):
            env_pool.extend(vec_env._envs)
        else:
            env_pool.append(vec_env)
    return env_pool


def evaluate(model, env_pool, num_meta_episodes, env_horizon, n_eval=10):
    """
    Evaluate the model on test environments.
    
    Note: Evaluation uses unnormalized rewards directly from the environment
    (not through VecNormalize) to get true performance metrics.
    """
    all_returns = []
    
    for env in env_pool[:n_eval]:
        # Create meta-env with single env in pool (fixed goal for this eval)
        # This is a fresh env without VecNormalize, so rewards are unnormalized
        meta_env = MetaEnv([env], num_meta_episodes, env_horizon)
        
        # Rollout
        obs = meta_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        episode_rewards = []
        current_ep_reward = 0
        
        for _ in range(meta_env.meta_horizon):
            action, lstm_states = model.predict(
                obs[None], state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, done, _ = meta_env.step(action[0])
            # reward here is unnormalized (raw from environment)
            current_ep_reward += reward
            episode_starts = np.array([done])
            
            # Track per-episode returns using internal done signal
            if obs[-1] > 0.5:  # prev_done
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0
            
            if done:
                break
        
        all_returns.append(episode_rewards)
    
    # Compute mean/std per episode across goals
    max_eps = max(len(r) for r in all_returns)
    padded = np.zeros((len(all_returns), max_eps))
    for i, r in enumerate(all_returns):
        padded[i, :len(r)] = r
    
    mean_returns = padded.mean(axis=0)
    std_returns = padded.std(axis=0)
    
    return mean_returns, std_returns, all_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL^2 Baseline")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="rl2")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)
    
    # Meta-learning
    parser.add_argument("--num_meta_episodes", type=int, default=4, help="Episodes per meta-episode")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Number of envs to create")
    
    # Training
    parser.add_argument("--total_timesteps", type=int, default=2000000)  # 2M (converges ~100K)
    parser.add_argument("--n_envs", type=int, default=16, help="Parallel meta-envs for PPO (VariBAD: 16)")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate (VariBAD: 7e-4)")
    parser.add_argument("--n_steps", type=int, default=60, help="Steps per update (VariBAD: 60)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (computed from n_minibatches)")
    parser.add_argument("--n_epochs", type=int, default=2, help="PPO epochs (VariBAD: 2)")
    parser.add_argument("--n_minibatches", type=int, default=4, help="Number of minibatches (VariBAD: 4)")
    
    # Evaluation
    parser.add_argument("--eval_freq", type=int, default=50000)
    parser.add_argument("--n_eval", type=int, default=10)
    parser.add_argument("--eval_meta_episodes", type=int, default=40, help="Episodes per meta-episode at eval time")
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="asteroid")
    parser.add_argument("--save_dir", type=str, default="results/rl2")

    # Advisor / BCPPO
    parser.add_argument("--use_advisor", action="store_true", help="Use Advisor with learned distance predictor")
    parser.add_argument("--use_bcppo", action="store_true", help="Use simple BC + PPO with decaying BC coefficient")
    parser.add_argument("--bc_decay", type=float, default=0.995, help="BC loss coefficient decay rate (BCPPO)")
    parser.add_argument("--advisor_alpha", type=float, default=4.0, help="Advisor weight scaling (Advisor)")
    parser.add_argument("--advisor_beta", type=float, default=0.1, help="Distance power: distance = (-log_prob)^beta (Advisor)")

    # Varibad
    parser.add_argument("--use_varibad", action="store_true")
    
    args = parse_with_config(parser)

    # Compute batch_size from n_minibatches if not specified
    if args.batch_size is None:
        args.batch_size = (args.n_envs * args.n_steps) // args.n_minibatches
    
    # Validate mutually exclusive options
    num_methods = sum([args.use_advisor, args.use_bcppo, args.use_varibad])
    assert num_methods <= 1, "Cannot use multiple methods: choose one of --use_advisor, --use_bcppo, --use_varibad"
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Save directory
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )
    
    # Create environments using create_env
    print(f"Creating environments: {args.env_name}")
    train_vec_envs, test_vec_envs, eval_vec_envs = create_env(
        args.env_name, args.dataset_size, n_envs=100  # batch size for creation
    )
    
    # Extract individual envs from VecEnvs
    train_env_pool = extract_env_pool(train_vec_envs)
    test_env_pool = extract_env_pool(test_vec_envs)
    eval_env_pool = extract_env_pool(eval_vec_envs)
    
    env_horizon = train_env_pool[0].horizon
    
    print(f"Train env pool: {len(train_env_pool)}, Test: {len(test_env_pool)}, Eval: {len(eval_env_pool)}")
    print(f"Env horizon: {env_horizon}")
    print(f"Training meta-episodes: {args.num_meta_episodes}, Eval meta-episodes: {args.eval_meta_episodes}")
    
    # Create vectorized meta-environment for training (no subprocess overhead)
    vec_env = MetaVecEnv(train_env_pool, args.n_envs, args.num_meta_episodes, env_horizon)
    
    # Wrap with reward normalization (VariBAD: norm_rew_for_policy=True)
    # Note: norm_obs=False because we handle obs augmentation manually
    # vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.95)
    # print("Reward normalization enabled (VecNormalize)")
    
    # Determine reward type for VariBAD (binary vs multiclass)
    reward_type = "multiclass" if "keydoor-markovian" in args.env_name else "binary"
    num_reward_classes = 3 if reward_type == "multiclass" else 2
    print(f"Reward type: {reward_type} (num_classes={num_reward_classes})")
    
    # Create model
    if args.use_advisor or args.use_bcppo:
        from train_advisor import AdvisorPPO, AdvisorPolicy
        
        # Policy architecture aligned with VariBAD
        advisor_policy_kwargs = dict(
            net_arch=dict(pi=[32], vf=[32]),
            lstm_hidden_size=64,
            n_lstm_layers=1,
            activation_fn=torch.nn.Tanh,
        )
        
        model = AdvisorPPO(
            AdvisorPolicy,
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.05,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=args.seed,
            policy_kwargs=advisor_policy_kwargs,
            # Advisor / BCPPO specific
            use_bcppo=args.use_bcppo,
            bc_decay=args.bc_decay,
            advisor_alpha=args.advisor_alpha,
            advisor_beta=args.advisor_beta,
        )
        mode = "BCPPO" if args.use_bcppo else "Advisor"
        print(f"Using {mode} mode (alpha={args.advisor_alpha}, beta={args.advisor_beta}, bc_decay={args.bc_decay})")
        
    elif args.use_varibad:
        from train_varibad import VariBADPPO, VariBADPolicy
        
        # VariBAD-aligned policy architecture
        varibad_policy_kwargs = dict(
            net_arch=dict(pi=[32], vf=[32]),  # One hidden layer of 32
            lstm_hidden_size=64,               # VariBAD: encoder_gru_hidden_size=64
            n_lstm_layers=1,
            activation_fn=torch.nn.Tanh,       # VariBAD: policy_activation_function=tanh
            latent_dim=32,                     # VariBAD: latent_dim=32
            decoder_hidden=32,                 # VariBAD: decoder layers [32, 32]
            reward_type=reward_type,           # binary or multiclass
            num_reward_classes=num_reward_classes,
        )
        
        model = VariBADPPO(
            VariBADPolicy,
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=2,                # VariBAD: ppo_num_epochs=2
            gamma=0.95,                # VariBAD: policy_gamma=0.95
            gae_lambda=0.95,           # VariBAD: policy_tau=0.95
            clip_range=0.05,           # VariBAD: ppo_clip_param=0.05
            ent_coef=0.01,             # VariBAD: policy_entropy_coef=0.01
            vf_coef=0.5,               # VariBAD: policy_value_loss_coef=0.5
            max_grad_norm=0.5,         # VariBAD: policy_max_grad_norm=0.5
            verbose=1,
            seed=args.seed,
            policy_kwargs=varibad_policy_kwargs,
            # VAE loss coefficients
            latent_dim=32,
            kl_coef=1.0,               # VariBAD: kl_weight=1.0
            reward_coef=1.0,           # VariBAD: rew_loss_coeff=1.0
            state_coef=1.0,            # VariBAD: state_loss_coeff=1.0
        )
    else:
        # VariBAD-aligned policy architecture: one hidden layer of 32 units
        policy_kwargs = dict(
            net_arch=dict(
                pi=[32],  # One hidden layer of 32 for policy (paper spec)
                vf=[32],  # One hidden layer of 32 for value
            ),
            lstm_hidden_size=64,  # VariBAD: encoder_gru_hidden_size=64
            n_lstm_layers=1,
            activation_fn=torch.nn.Tanh,  # VariBAD: policy_activation_function=tanh
        )
        
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.95,           # VariBAD: policy_gamma=0.95
            gae_lambda=0.95,      # VariBAD: policy_tau=0.95
            clip_range=0.05,      # VariBAD: ppo_clip_param=0.05 (smaller than SB3 default 0.2)
            ent_coef=0.01,        # VariBAD: policy_entropy_coef=0.01
            vf_coef=0.5,          # VariBAD: policy_value_loss_coef=0.5
            max_grad_norm=0.5,    # VariBAD: policy_max_grad_norm=0.5
            verbose=1,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
        )
    
    print(f"Model created. Training for {args.total_timesteps} timesteps...")
    
    # Training loop with periodic evaluation
    timesteps_done = 0
    eval_results = []
    
    while timesteps_done < args.total_timesteps:
        # Train for eval_freq steps
        steps_to_train = min(args.eval_freq, args.total_timesteps - timesteps_done)
        
        callbacks = [WandbCallback()] if args.log_wandb else []
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False, callback=callbacks)
        timesteps_done += steps_to_train
        
        # Evaluate on test envs (use eval_meta_episodes=40, not training num_meta_episodes=4)
        print(f"\nEvaluating at {timesteps_done} timesteps...")
        mean_returns, std_returns, _ = evaluate(
            model, eval_env_pool, args.eval_meta_episodes, env_horizon, args.n_eval
        )
        
        eval_results.append({
            "timesteps": timesteps_done,
            "mean_returns": mean_returns,
            "std_returns": std_returns,
        })
        
        print(f"  Final episode return: {mean_returns[-1]:.3f} ± {std_returns[-1]:.3f}")
        
        if args.log_wandb:
            wandb.log({
                "eval/final_return": mean_returns[-1],
                "eval/mean_return": np.mean(mean_returns),
                "timesteps": timesteps_done,
            })
            
            # Log per-episode returns
            for ep_idx, (mean_ret, std_ret) in enumerate(zip(mean_returns, std_returns)):
                wandb.log({f"eval/episode_{ep_idx}_return": mean_ret, "timesteps": timesteps_done})
    
    # Save final model and results
    model.save(os.path.join(save_dir, "final_model"))
    
    with open(os.path.join(save_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(eval_results, f)
    
    # Final evaluation (use eval_meta_episodes=40 for full adaptation curve)
    mean_returns, std_returns, all_returns = evaluate(
        model, eval_env_pool, args.eval_meta_episodes, env_horizon, len(eval_env_pool)
    )
    
    np.savez(
        os.path.join(save_dir, "eval_returns.npz"),
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(len(mean_returns))
    ax.plot(episodes, mean_returns, label='Mean Return', linewidth=2)
    ax.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(f'RL^2 Eval Returns - {args.env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "eval_returns.png"), dpi=150, bbox_inches='tight')
    
    if args.log_wandb:
        wandb.log({"eval/returns_plot": wandb.Image(fig)})
    
    plt.close(fig)
    
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    if args.log_wandb:
        wandb.finish()
