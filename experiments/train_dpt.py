"""DPT baseline: offline supervised pretraining of a Decision Transformer.

Trains on an offline dataset of randomly collected in-context interactions,
predicting the clairvoyant expert action at each query state (Decision-Pretrained
Transformer, Lee et al., 2023). Unlike ASTEROID, contexts are gathered by a fixed
random roll-in policy with no on-policy iterations, so DPT relies on broad
coverage of the history space to learn in-context exploration.

Shares the environment, dataset, model, and training loop with ASTEROID.
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse

from experiments.config import parse_with_config
import os
import pickle
import random

import numpy as np
import torch
import wandb

from environments.create_envs import create_env
from environments.rollout_policy import get_rollout_policy
from datasets.collect_data import get_dagger_dataset
from datasets.dataset import collate_fn
from experiments.eval_policy import evaluate_policy_on_envs
from experiments.train_asteroid import get_optimizer_scheduler, train_step
from models import DecisionTransformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPT (offline in-context pretraining)")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="dpt")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--n_envs", type=int, default=10000)
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Episodes of random context per trajectory")

    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=40)

    # Model
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--eval_interval", type=float, default=0.1)
    parser.add_argument("--save_interval", type=float, default=0.1)

    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="asteroid")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Paths
    parser.add_argument("--save_dir", type=str, default="results/dpt")

    args = parse_with_config(parser)

    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Environments
    print(f"Creating environments: {args.env_name}")
    train_envs, test_envs, eval_envs = create_env(args.env_name, args.dataset_size, args.n_envs)
    state_dim = train_envs[0]._envs[0].state_dim
    action_dim = train_envs[0]._envs[0].action_dim
    env_horizon = train_envs[0]._envs[0].horizon

    # A DPT context spans several episodes of random interaction.
    horizon = env_horizon * args.n_episodes
    model_args = {
        "horizon": horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_embd": 128,
        "dropout": args.dropout,
        "shuffle": True,
        "test": False,
        "continuous_action": False,
        "gmm_heads": 1,
    }
    with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
        pickle.dump(model_args, f)

    model = DecisionTransformer(model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Offline data collection with a random roll-in policy and expert labels.
    random_policy = get_rollout_policy("random")
    train_dataset, test_dataset = get_dagger_dataset(
        train_envs, test_envs, random_policy, horizon
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    total_steps = len(train_dataset) // args.batch_size * args.num_epochs
    optimizer, scheduler = get_optimizer_scheduler(model, total_steps, args.lr, args.warmup_ratio)

    # Single offline training phase (reuses the ASTEROID training loop).
    model = train_step(
        0, model, optimizer, scheduler, train_loader, test_loader,
        save_dir, args, device, action_dim, env_horizon,
    )

    # Evaluate the in-context policy over eval_episodes test-time episodes.
    eval_policy = get_rollout_policy(
        "decision_transformer",
        model=model,
        context_horizon=horizon,
        env_horizon=env_horizon,
        context_accumulation=False,
        sliding_window=True,
    )
    eval_results = evaluate_policy_on_envs(
        eval_envs=eval_envs,
        policy=eval_policy,
        eval_horizon=args.eval_episodes * env_horizon,
        env_horizon=env_horizon,
        save_dir=os.path.join(save_dir, "eval"),
        env_name=args.env_name,
        plot=True,
    )

    if args.log_wandb:
        for ep_idx, mean_ret in enumerate(eval_results["mean_returns"]):
            wandb.log({f"eval/episode_{ep_idx}_return": mean_ret})
        wandb.log({"eval/final_return": eval_results["mean_returns"][-1]})

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print(f"\nTraining complete! Results saved to {save_dir}")

    if args.log_wandb:
        wandb.finish()
