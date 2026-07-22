"""Darkroom: sparse-reward 2D goal-search gridworld (single + vectorized)."""

import gym
import numpy as np

from environments.base_env import BaseEnv


class DarkroomEnv(BaseEnv):
    """
    Darkroom environment: Navigate a grid to find a hidden goal.
    
    Actions: 0=right, 1=left, 2=down, 3=up, 4=stay
    Reward: 1 when at goal, 0 otherwise.
    """

    def __init__(self, dim, goal, horizon):
        self.dim = dim
        self.goal = np.array(goal)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 5
        self.observation_space = gym.spaces.Box(
            low=0, high=dim - 1, shape=(self.state_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.action_dim)

    def sample_state(self):
        return np.random.randint(0, self.dim, 2).astype(float)

    def sample_action(self):
        action = np.zeros(self.action_space.n)
        action[np.random.randint(0, self.action_space.n)] = 1
        return action

    def reset(self):
        self.current_step = 0
        self.state = np.array([0, 0], dtype=float)
        return self.state.copy()

    def transit(self, state, action):
        action_idx = np.argmax(action)
        state = np.array(state)
        
        # Apply movement: 0=right, 1=left, 2=down, 3=up, 4=stay
        if action_idx == 0:
            state[0] += 1
        elif action_idx == 1:
            state[0] -= 1
        elif action_idx == 2:
            state[1] += 1
        elif action_idx == 3:
            state[1] -= 1
        
        state = np.clip(state, 0, self.dim - 1)
        reward = float(np.all(state == self.goal))
        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, reward = self.transit(self.state, action)
        self.current_step += 1
        done = self.current_step >= self.horizon
        return self.state.copy(), reward, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        """Return optimal action to reach goal."""
        if state[0] < self.goal[0]:
            action_idx = 0  # right
        elif state[0] > self.goal[0]:
            action_idx = 1  # left
        elif state[1] < self.goal[1]:
            action_idx = 2  # down
        elif state[1] > self.goal[1]:
            action_idx = 3  # up
        else:
            action_idx = 4  # stay
        
        action = np.zeros(self.action_space.n)
        action[action_idx] = 1
        return action


class DarkroomEnvVec(BaseEnv):
    """Vectorized Darkroom environment for parallel execution."""

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self._goals = np.array([env.goal for env in envs])
        self._dim = envs[0].dim
        self.horizon = envs[0].horizon
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    @property
    def state_dim(self):
        return self._envs[0].state_dim

    @property
    def action_dim(self):
        return self._envs[0].action_dim

    def sample_state(self):
        return np.random.randint(0, self._dim, (self._num_envs, 2)).astype(float)

    def sample_action(self):
        actions = np.zeros((self._num_envs, self.action_dim))
        actions[np.arange(self._num_envs), np.random.randint(0, self.action_dim, self._num_envs)] = 1
        return actions

    def reset(self):
        self.current_step = np.zeros(self._num_envs, dtype=int)
        self.states = np.zeros((self._num_envs, 2), dtype=float)
        return self.states.copy()

    def transit(self, states, actions):
        action_idxs = np.argmax(actions, axis=1)
        
        # Apply movements: 0=right, 1=left, 2=down, 3=up
        states[:, 0] += (action_idxs == 0).astype(float)
        states[:, 0] -= (action_idxs == 1).astype(float)
        states[:, 1] += (action_idxs == 2).astype(float)
        states[:, 1] -= (action_idxs == 3).astype(float)
        
        states = np.clip(states, 0, self._dim - 1)
        rewards = (np.linalg.norm(states - self._goals, axis=1) < 1e-5).astype(float)
        return states, rewards

    def step(self, actions):
        if np.any(self.current_step >= self.horizon):
            raise ValueError("Episode has already ended for some environments")

        self.states, rewards = self.transit(self.states, actions)
        self.current_step += 1
        dones = self.current_step >= self.horizon
        return self.states.copy(), rewards, dones, {}

    def opt_action(self, states):
        actions = np.array([env.opt_action(state) for env, state in zip(self._envs, states)])
        return actions
