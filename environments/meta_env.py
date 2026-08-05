"""
MetaEnv wrapper for RL^2.

Wraps a base environment to create a meta-learning environment where:
- Goals are resampled every num_meta_episodes episodes
- Observations are augmented with (reward, done) from previous step
- Done is only True at the end of all meta-episodes
"""

import copy

import gym
import gymnasium as gymnasium
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class MetaEnv(gym.Env):
    """
    Meta-environment wrapper for RL^2.
    
    The agent receives augmented observations: [obs, prev_reward, prev_done]
    This allows the agent to learn to adapt based on reward history.
    
    Args:
        env_pool: List of base environments to sample from (each has its own goal)
        num_meta_episodes: Number of episodes per meta-episode
        env_horizon: Steps per episode (if None, uses env_pool[0].horizon)
    """
    
    def __init__(self, env_pool, num_meta_episodes, env_horizon=None):
        self.env_pool = env_pool
        self.num_meta_episodes = num_meta_episodes
        self.env_horizon = env_horizon or getattr(env_pool[0], 'horizon', 100)
        
        # Current active environment
        self.current_env = env_pool[0]
        
        # Total steps in a meta-episode
        self.meta_horizon = num_meta_episodes * self.env_horizon
        
        # Augmented observation space: [obs, prev_reward, prev_done]
        base_obs_space = env_pool[0].observation_space
        low = np.concatenate([base_obs_space.low, [0.0, 0.0]])
        high = np.concatenate([base_obs_space.high, [1.0, 1.0]])
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Action space from base env
        # self.action_space = env_pool[0].action_space
        self.action_space = gymnasium.spaces.Discrete(env_pool[0].action_space.n)
        
        # Internal state
        self.current_step = 0
        self.current_episode = 0
        self.episode_step = 0
        self.prev_reward = 0.0
        self.prev_done = 0.0
        
    def _get_augmented_obs(self, obs):
        """Augment observation with previous reward and done."""
        return np.concatenate([obs, [self.prev_reward, self.prev_done]]).astype(np.float32)
    
    def _sample_env(self):
        """Sample a new environment from the pool (deepcopy to avoid sharing)."""
        idx = np.random.randint(len(self.env_pool))
        self.current_env = copy.deepcopy(self.env_pool[idx])
    
    def reset(self):
        """Reset the meta-environment."""
        self.current_step = 0
        self.current_episode = 0
        self.episode_step = 0
        self.prev_reward = 0.0
        self.prev_done = 0.0
        
        # Sample new env (with its goal) for meta-episode
        self._sample_env()
        obs = self.current_env.reset()
        
        return self._get_augmented_obs(obs)
    
    def step(self, action):
        """
        Step the environment.
        
        Returns done=True only at the end of all meta-episodes.
        Internal episode boundaries are signaled via prev_done in observation.
        """
        # Convert discrete action to one-hot if needed
        if isinstance(action, (int, np.integer)):
            action_onehot = np.zeros(self.action_space.n)
            action_onehot[action] = 1
            action = action_onehot
        
        obs, reward, episode_done, info = self.current_env.step(action)
        
        self.current_step += 1
        self.episode_step += 1
        
        # Check if episode ended (either by done or horizon)
        if episode_done or self.episode_step >= self.env_horizon:
            self.current_episode += 1
            self.episode_step = 0
            internal_done = 1.0
            
            # Reset base env for next episode within meta-episode
            if self.current_episode < self.num_meta_episodes:
                obs = self.current_env.reset()
        else:
            internal_done = 0.0
        
        # Meta-episode done only after all episodes
        meta_done = self.current_episode >= self.num_meta_episodes
        
        # Update previous reward/done for next observation
        self.prev_reward = reward
        self.prev_done = internal_done
        
        augmented_obs = self._get_augmented_obs(obs)
        
        return augmented_obs, reward, meta_done, info


class MetaVecEnv(VecEnv):
    """
    Vectorized meta-environment for RL^2.
    
    Inherits from stable-baselines3 VecEnv for full compatibility.
    Each parallel env samples from a shared pool of base environments.
    
    Args:
        env_pool: List of base environments to sample from
        num_parallel: Number of parallel meta-environments
        num_meta_episodes: Number of episodes per meta-episode
        env_horizon: Steps per episode
    """
    
    def __init__(self, env_pool, num_parallel, num_meta_episodes, env_horizon=None):
        self.env_pool = env_pool
        self.num_meta_episodes = num_meta_episodes
        self.env_horizon = env_horizon or getattr(env_pool[0], 'horizon', 100)
        
        # Current active environment for each parallel env
        self.current_envs = [None] * num_parallel
        
        # Meta horizon
        self.meta_horizon = num_meta_episodes * self.env_horizon
        
        # Augmented observation space
        base_obs_space = env_pool[0].observation_space
        low = np.concatenate([base_obs_space.low, [0.0, 0.0]])
        high = np.concatenate([base_obs_space.high, [1.0, 1.0]])
        observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        
        # action_space = env_pool[0].action_space
        action_space = gymnasium.spaces.Discrete(env_pool[0].action_space.n)
        
        # Initialize parent VecEnv
        super().__init__(num_parallel, observation_space, action_space)
        
        # Internal state for each env
        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episodes = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.prev_dones = np.zeros(self.num_envs, dtype=np.float32)
        
        # For sb3 compatibility
        self._actions = None

        self._last_expert_actions = np.zeros((self.num_envs, self.action_space.n), dtype=np.int64)
        
    def _get_augmented_obs(self, obs, rewards=None, dones=None):
        """Augment observations with previous rewards and dones."""
        if rewards is None:
            rewards = self.prev_rewards
        if dones is None:
            dones = self.prev_dones
        # Handle single obs case
        if obs.ndim == 1:
            return np.concatenate([obs, [rewards, dones]]).astype(np.float32)
        return np.concatenate([
            obs,
            np.atleast_1d(rewards)[:, None],
            np.atleast_1d(dones)[:, None]
        ], axis=1).astype(np.float32)
    
    def _sample_env(self, idx):
        """Sample a new environment for a specific index (deepcopy to avoid sharing)."""
        env_idx = np.random.randint(len(self.env_pool))
        # Deepcopy to ensure each slot has independent env state
        self.current_envs[idx] = copy.deepcopy(self.env_pool[env_idx])
    
    def _reset_idx(self, idx):
        """Reset a single environment at index."""
        self.current_steps[idx] = 0
        self.current_episodes[idx] = 0
        self.episode_steps[idx] = 0
        self.prev_rewards[idx] = 0.0
        self.prev_dones[idx] = 0.0
        self._sample_env(idx)
        obs = self.current_envs[idx].reset()
        if hasattr(self.current_envs[idx], "have_key"):
            self._last_expert_actions[idx] = self.current_envs[idx].opt_action(obs, self.current_envs[idx].have_key)
        else:
            self._last_expert_actions[idx] = self.current_envs[idx].opt_action(obs)
        return obs
    
    def reset(self):
        """Reset all environments."""
        obs_list = []
        for i in range(self.num_envs):
            obs_list.append(self._reset_idx(i))
        obs = np.stack(obs_list)
        return self._get_augmented_obs(obs)
    
    def step_async(self, actions):
        """Store actions for async step."""
        self._actions = actions
    
    def step_wait(self):
        """Execute stored actions."""
        actions = self._actions
        
        # Convert discrete actions to one-hot if needed
        if actions.ndim == 1:
            actions_onehot = np.zeros((self.num_envs, self.action_space.n))
            actions_onehot[np.arange(self.num_envs), actions] = 1
            actions = actions_onehot
        
        obs_list = []
        rewards = np.zeros(self.num_envs)
        internal_dones = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        for i, (env, action) in enumerate(zip(self.current_envs, actions)):
            obs, reward, done, _ = env.step(action)
            rewards[i] = reward

            infos[i]["expert_action"] = self._last_expert_actions[i]
            
            self.current_steps[i] += 1
            self.episode_steps[i] += 1
            
            # Check episode end
            if done or self.episode_steps[i] >= self.env_horizon:
                self.current_episodes[i] += 1
                self.episode_steps[i] = 0
                internal_dones[i] = 1.0
                
                # Reset for next episode within meta-episode
                if self.current_episodes[i] < self.num_meta_episodes:
                    obs = env.reset()
            
            if hasattr(env, "have_key"):
                self._last_expert_actions[i] = env.opt_action(obs, env.have_key)
            else:
                self._last_expert_actions[i] = env.opt_action(obs)
            
            # Check meta-episode end
            if self.current_episodes[i] >= self.num_meta_episodes:
                dones[i] = True
                # Auto-reset for sb3 (store terminal obs in info)
                infos[i]['terminal_observation'] = self._get_augmented_obs(
                    obs, rewards=rewards[i], dones=internal_dones[i]
                )
                obs = self._reset_idx(i)
            
            obs_list.append(obs)
        
        obs = np.stack(obs_list)
        self.prev_rewards = rewards.copy()
        self.prev_dones = internal_dones.copy()
        
        return self._get_augmented_obs(obs), rewards, dones, infos
    
    def close(self):
        """Close environments."""
        pass
    
    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if envs are wrapped."""
        return [False] * self.num_envs
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call method on envs."""
        return []
    
    def get_attr(self, attr_name, indices=None):
        """Get attribute from envs."""
        return [getattr(env, attr_name, None) for env in self.current_envs]
    
    def set_attr(self, attr_name, value, indices=None):
        """Set attribute on envs."""
        pass
