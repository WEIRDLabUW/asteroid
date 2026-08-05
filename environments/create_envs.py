"""
Environment creation utilities.

Supported environments:
1. darkroom-easy: 10x10 grid, horizon=100
2. darkroom-hard: 20x20 grid, horizon=200
3. keydoor-nonmarkovian: 5x5 grid, one-time rewards
4. keydoor-markovian: 5x5 grid, continuous rewards after key/door
5. navigation-episodic: 2D continuous, horizon=10, resets between episodes
6. navigation-nonepisodic: 2D continuous, horizon=10, no reset between episodes
"""

from itertools import combinations

import numpy as np

from environments.darkroom_env import DarkroomEnv, DarkroomEnvVec
from environments.keydoor_env import KeyDoorEnv, KeyDoorVecEnv
from environments.navigation_env import NavigationEnv, NavigationVecEnv


def _batch_envs(envs, vec_env_class, n_envs):
    """Helper to batch single envs into vectorized env batches."""
    return [
        vec_env_class(envs[i : i + n_envs])
        for i in range(0, len(envs), n_envs)
    ]


def create_darkroom_env(env_name, dataset_size, n_envs):
    """
    Create Darkroom environments.
    
    Args:
        env_name: "darkroom-easy" (10x10, H=100) or "darkroom-hard" (20x20, H=200)
        dataset_size: Number of environments to create
        n_envs: Batch size for vectorized environments
    
    Returns:
        train_envs, test_envs, eval_envs: Lists of vectorized environments
    """
    # if env_name == "darkroom-easy":
    if "easy" in env_name:
        dim, horizon = 10, 100
    elif "hard" in env_name:
        dim, horizon = 20, 200
    elif "easy-small" in env_name:
        dim, horizon = 5, 50
    else:
        raise ValueError(f"Unknown darkroom variant: {env_name} - should be easy or hard")

    # Generate all grid positions as goals
    goals = np.array([[i, j] for i in range(dim) for j in range(dim)])
    np.random.RandomState(seed=0).shuffle(goals)
    
    # 80/20 train/test split
    split_idx = int(0.8 * len(goals))
    train_goals = goals[:split_idx]
    test_goals = goals[split_idx:]
    
    # Repeat goals to match dataset_size
    n_repeats = max(1, dataset_size // len(goals))
    train_goals = np.repeat(train_goals, n_repeats, axis=0)
    test_goals = np.repeat(test_goals, n_repeats, axis=0)
    
    # Eval goals: use test goals, ensure at least 100
    eval_factor = max(1, 100 // len(goals[split_idx:]))
    eval_goals = np.tile(goals[split_idx:], (eval_factor, 1))

    # Create single environments
    train_envs = [DarkroomEnv(dim, goal, horizon) for goal in train_goals]
    test_envs = [DarkroomEnv(dim, goal, horizon) for goal in test_goals]
    eval_envs = [DarkroomEnv(dim, goal, horizon) for goal in eval_goals]

    # Batch into vectorized environments
    train_envs = _batch_envs(train_envs, DarkroomEnvVec, n_envs)
    test_envs = _batch_envs(test_envs, DarkroomEnvVec, n_envs)
    eval_envs = _batch_envs(eval_envs, DarkroomEnvVec, n_envs)

    return train_envs, test_envs, eval_envs


def create_keydoor_env(env_name, dataset_size, n_envs):
    """
    Create Key-Door environments.
    
    Args:
        env_name: "keydoor-nonmarkovian" or "keydoor-markovian"
        dataset_size: Number of environments to create
        n_envs: Batch size for vectorized environments
    
    Returns:
        train_envs, test_envs, eval_envs: Lists of vectorized environments
    """
    dim = 5
    horizon = 50
    markovian = "nonmarkovian" not in env_name

    # Generate all grid positions
    locations = np.array([[i, j] for i in range(dim) for j in range(dim)])
    location_idxs = np.arange(len(locations))
    
    # Generate all unique key-door pairs
    key_door_pairs = np.array(list(combinations(location_idxs, 2)))
    np.random.RandomState(seed=0).shuffle(key_door_pairs)
    
    # 80/20 train/test split
    split_idx = int(0.8 * len(key_door_pairs))
    train_pairs = key_door_pairs[:split_idx]
    test_pairs = key_door_pairs[split_idx:]
    
    # Repeat pairs to match dataset_size
    n_repeats = max(1, dataset_size // len(key_door_pairs))
    train_pairs = np.repeat(train_pairs, n_repeats, axis=0)
    test_pairs = np.repeat(test_pairs, n_repeats, axis=0)
    
    # Eval pairs: use test pairs, ensure at least 100
    eval_factor = max(1, 100 // len(key_door_pairs[split_idx:]))
    eval_pairs = np.tile(key_door_pairs[split_idx:], (eval_factor, 1))

    # Create single environments
    train_envs = [
        KeyDoorEnv(dim, locations[k], locations[d], horizon, markovian)
        for k, d in train_pairs
    ]
    test_envs = [
        KeyDoorEnv(dim, locations[k], locations[d], horizon, markovian)
        for k, d in test_pairs
    ]
    eval_envs = [
        KeyDoorEnv(dim, locations[k], locations[d], horizon, markovian)
        for k, d in eval_pairs
    ]

    # Batch into vectorized environments
    train_envs = _batch_envs(train_envs, KeyDoorVecEnv, n_envs)
    test_envs = _batch_envs(test_envs, KeyDoorVecEnv, n_envs)
    eval_envs = _batch_envs(eval_envs, KeyDoorVecEnv, n_envs)

    return train_envs, test_envs, eval_envs


def create_navigation_env(env_name, dataset_size, n_envs):
    """
    Create 2D Navigation environments.
    
    Args:
        env_name: "navigation-episodic" or "navigation-nonepisodic"
        dataset_size: Number of environments to create
        n_envs: Batch size for vectorized environments
    
    Returns:
        train_envs, test_envs, eval_envs: Lists of vectorized environments
    """
    radius = 1.0
    horizon = 20
    goal_tolerance = 0.2
    reset_free = "nonepisodic" in env_name

    # Generate 100 goals on semi-circle
    n_goals = 100
    angles = np.linspace(0, np.pi, n_goals)
    goals = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
    np.random.RandomState(seed=0).shuffle(goals)
    
    # 80/20 train/test split
    split_idx = int(0.8 * len(goals))
    train_goals = goals[:split_idx]
    test_goals = goals[split_idx:]
    
    # Repeat goals to match dataset_size
    n_repeats = max(1, dataset_size // len(goals))
    train_goals = np.repeat(train_goals, n_repeats, axis=0)
    test_goals = np.repeat(test_goals, n_repeats, axis=0)
    
    # Eval goals: use test goals, ensure at least 100
    eval_factor = max(1, 100 // len(goals[split_idx:]))
    eval_goals = np.tile(goals[split_idx:], (eval_factor, 1))

    # Create single environments
    train_envs = [
        NavigationEnv(radius, goal, horizon, reset_free, goal_tolerance)
        for goal in train_goals
    ]
    test_envs = [
        NavigationEnv(radius, goal, horizon, reset_free, goal_tolerance)
        for goal in test_goals
    ]
    eval_envs = [
        NavigationEnv(radius, goal, horizon, reset_free, goal_tolerance)
        for goal in eval_goals
    ]

    # Batch into vectorized environments
    train_envs = _batch_envs(train_envs, NavigationVecEnv, n_envs)
    test_envs = _batch_envs(test_envs, NavigationVecEnv, n_envs)
    eval_envs = _batch_envs(eval_envs, NavigationVecEnv, n_envs)

    return train_envs, test_envs, eval_envs


def create_env(env_name, dataset_size, n_envs):
    """
    Create environments based on name.
    
    Args:
        env_name: Environment identifier
        dataset_size: Number of environments to create  
        n_envs: Batch size for vectorized environments
    
    Returns:
        train_envs, test_envs, eval_envs: Lists of vectorized environments
    """
    if "darkroom" in env_name:
        return create_darkroom_env(env_name, dataset_size, n_envs)
    elif "keydoor" in env_name:
        return create_keydoor_env(env_name, dataset_size, n_envs)
    elif "navigation" in env_name:
        return create_navigation_env(env_name, dataset_size, n_envs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def test_all_envs():
    """Test that optimal policies achieve positive reward in all environments."""
    env_names = [
        "navigation-episodic",
        "navigation-nonepisodic",
        "darkroom-easy",
        "darkroom-hard",
        "keydoor-nonmarkovian",
        "keydoor-markovian",
    ]
    
    for name in env_names:
        train_envs, test_envs, eval_envs = create_env(name, 1, 1)
        print(f"Testing {name}...")
        for env in train_envs:
            obs = env.reset()
            total_reward = 0
            done = False
            while not done:
                if "keydoor" in name:
                    action = env.opt_action(obs, env.have_keys)
                else:
                    action = env.opt_action(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
        print(f"  Total reward: {total_reward}")
        assert total_reward > 0, f"Optimal policy failed in {name}"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_all_envs()
