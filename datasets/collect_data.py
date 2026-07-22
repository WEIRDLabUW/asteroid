"""
Data collection utilities for training.

Functions for:
- DAgger-style data collection with expert supervision
- Saving/loading trajectory data
- Merging datasets
- Policy evaluation
"""

import pickle
import numpy as np
import tqdm


def dagger_rollout(env, rollout_policy, horizon):
    """
    Collect a rollout with expert action labels (DAgger-style).
    
    Args:
        env: Vectorized environment
        rollout_policy: Policy to collect actions from
        horizon: Number of steps to collect
    
    Returns:
        Dictionary with:
        - states: (n_envs, horizon, state_dim)
        - actions: (n_envs, horizon, action_dim) - policy actions
        - expert_actions: (n_envs, horizon, action_dim) - expert supervision
        - rewards: (n_envs, horizon)
        - dones: (n_envs, horizon)
    """
    rollout_policy.set_env(env)
    state = env.reset()
    rollout_policy.reset()
    n_envs = env.num_envs
    
    states = []
    actions = []
    expert_actions = []
    rewards = []
    dones_list = []

    for t in range(horizon):
        # Get policy action
        action = rollout_policy.get_action(state)
        
        # Clip continuous actions if needed
        if hasattr(rollout_policy, "continuous_action") and rollout_policy.continuous_action:
            if hasattr(env, "action_space"):
                action = np.clip(action, env.action_space.low, env.action_space.high)
        
        # Get expert action for supervision
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
        
        # Update policy context
        rollout_policy.update_context(state, action, reward, done)
        
        # Handle episode resets
        if np.any(done):
            next_state = env.reset()
        
        state = next_state

    # Stack arrays
    data = {
        "states": np.stack(states, axis=1),
        "actions": np.stack(actions, axis=1),
        "expert_actions": np.stack(expert_actions, axis=1),
        "rewards": np.stack(rewards, axis=1),
        "dones": np.stack(dones_list, axis=1),
    }
    
    # Verify shapes
    assert data["states"].shape == (n_envs, horizon, env.state_dim)
    assert data["actions"].shape == (n_envs, horizon, env.action_dim)
    assert data["expert_actions"].shape == (n_envs, horizon, env.action_dim)
    assert data["rewards"].shape == (n_envs, horizon)
    assert data["dones"].shape == (n_envs, horizon)
    
    return data


def get_dagger_data(envs, rollout_policy, horizon):
    """
    Collect DAgger data from multiple environments.
    
    Args:
        envs: List of vectorized environments
        rollout_policy: Policy for data collection
        horizon: Steps per environment
    
    Returns:
        List of trajectory dictionaries
    """
    trajs = []
    for env in tqdm.tqdm(envs, desc="Collecting dagger data"):
        data = dagger_rollout(env, rollout_policy, horizon)
        n_envs = env.num_envs
        
        for k in range(n_envs):
            traj = {
                "states": data["states"][k],
                "actions": data["actions"][k],
                "expert_actions": data["expert_actions"][k],
                "rewards": data["rewards"][k],
                "dones": data["dones"][k],
                "goal": env._envs[k].goal,
            }
            trajs.append(traj)
    return trajs


def get_dagger_dataset(train_envs, test_envs, rollout_policy, horizon):
    """
    Create train and test datasets using DAgger-style collection.
    
    Args:
        train_envs: List of training environments
        test_envs: List of test environments  
        rollout_policy: Policy for data collection
        horizon: Steps per environment
    
    Returns:
        train_dataset, test_dataset: SequenceDataset instances
    """
    from datasets.dataset import SequenceDataset

    train_trajs = get_dagger_data(train_envs, rollout_policy, horizon)
    test_trajs = get_dagger_data(test_envs, rollout_policy, horizon)
    
    config = {
        "horizon": horizon, 
        "store_gpu": False, 
        "state_dim": train_envs[0].state_dim, 
        "action_dim": train_envs[0].action_dim
    }
    
    train_dataset = SequenceDataset(train_trajs, {**config, "shuffle": True})
    test_dataset = SequenceDataset(test_trajs, {**config, "shuffle": False})
    
    return train_dataset, test_dataset


def merge_sequence_datasets(dataset1, dataset2):
    """
    Merge two SequenceDatasets.
    
    Args:
        dataset1: First dataset
        dataset2: Second dataset
    
    Returns:
        Merged SequenceDataset
    """
    from datasets.dataset import SequenceDataset
    merged_trajs = dataset1.trajs + dataset2.trajs
    return SequenceDataset(merged_trajs, dataset1.config)


def merge_trajs(trajs):
    """
    Merge list of trajectory dicts into single dict with stacked arrays.
    
    Args:
        trajs: List of trajectory dictionaries
    
    Returns:
        Dictionary with stacked numpy arrays
    """
    merged = {
        "states": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "expert_actions": [],
    }
    
    for traj in trajs:
        merged["states"].append(traj["states"])
        merged["actions"].append(traj["actions"])
        merged["rewards"].append(traj["rewards"])
        merged["dones"].append(traj["dones"])
        merged["expert_actions"].append(traj["expert_actions"])
    
    for key in merged:
        merged[key] = np.stack(merged[key], axis=0)
    
    return merged


def save_dagger_data(trajs, save_path):
    """
    Save trajectory data to disk.
    
    Args:
        trajs: List of trajectory dicts or merged dict
        save_path: Path to save pickle file
    
    Returns:
        Merged trajectory dict
    """
    if isinstance(trajs, list):
        trajs = merge_trajs(trajs)
    with open(save_path, 'wb') as f:
        pickle.dump(trajs, f)
    return trajs


def load_data(load_path):
    """
    Load trajectory data from disk.
    
    Args:
        load_path: Path to pickle file
    
    Returns:
        Trajectory dictionary
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def evaluate_policy(envs, policy, horizon, env_horizon):
    """
    Evaluate a policy and compute episode returns.
    
    Args:
        envs: List of evaluation environments
        policy: Policy to evaluate
        horizon: Total steps to run
        env_horizon: Steps per episode (for computing returns)
    
    Returns:
        Dictionary with:
        - episode_returns: List of episode returns
        - step_rewards: List of per-step rewards
        - mean_return: Mean episode return
        - std_return: Std of episode returns
    """
    all_episode_returns = []
    all_step_rewards = []
    
    for env in tqdm.tqdm(envs, desc="Evaluating"):
        data = dagger_rollout(env, policy, horizon)
        rewards = data["rewards"]
        dones = data["dones"]
        
        n_envs = rewards.shape[0]
        for i in range(n_envs):
            episode_reward = 0
            for t in range(horizon):
                episode_reward += rewards[i, t]
                all_step_rewards.append(rewards[i, t])
                
                if dones[i, t] or (t + 1) % env_horizon == 0:
                    all_episode_returns.append(episode_reward)
                    episode_reward = 0
    
    return {
        "episode_returns": all_episode_returns,
        "step_rewards": all_step_rewards,
        "mean_return": np.mean(all_episode_returns) if all_episode_returns else 0,
        "std_return": np.std(all_episode_returns) if all_episode_returns else 0,
    }
