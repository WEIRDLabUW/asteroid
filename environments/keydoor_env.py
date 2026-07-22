"""Key-Door gridworld: find a hidden key, then reach a hidden door (single + vectorized)."""

import gym
import numpy as np

from environments.base_env import BaseEnv


class KeyDoorEnv(BaseEnv):
    """
    Key-Door environment: Navigate a grid, pick up key, then open door.
    
    Actions: 0=right, 1=left, 2=down, 3=up, 4=stay
    
    Rewards:
    - Non-Markovian: +1 once when picking up key, +1 once when reaching door (with key)
    - Markovian: +1 every step after picking up key, +1 additional every step after reaching door
    """

    def __init__(self, dim, key, door, horizon, markovian):
        self.dim = dim
        self.key = np.array(key)
        self.door = np.array(door)
        self.horizon = horizon
        self.markovian = markovian
        self.state_dim = 2
        self.action_dim = 5
        self.goal = np.concatenate([key, door])
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

    def sample_flags(self, state):
        """Sample random key/door possession flags for data collection."""
        have_key = np.random.random() > 0.5
        have_door = have_key and np.random.random() > 0.5
        return have_key, have_door

    def reset(self):
        self.current_step = 0
        self.state = np.array([0, 0], dtype=float)
        self.have_key = False
        self.have_door = False
        return self.state.copy()

    def transit(self, state, action, have_key, have_door):
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

        # Check key pickup
        at_key = np.all(state == self.key) and not have_key
        have_key = at_key or have_key
        
        # Check door reach (only counts if have key)
        at_door = have_key and np.all(state == self.door) and not have_door
        have_door = at_door or have_door

        # Compute reward
        if self.markovian:
            reward = float(have_key) + float(have_door)
        else:
            reward = float(at_key) + float(at_door)

        return state, reward, have_key, have_door

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, reward, self.have_key, self.have_door = self.transit(
            self.state, action, self.have_key, self.have_door
        )
        self.current_step += 1
        done = self.current_step >= self.horizon
        return self.state.copy(), reward, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state, have_key):
        """Return optimal action to reach key (if not have) or door (if have key)."""
        goal = self.door if have_key else self.key
        
        if state[0] < goal[0]:
            action_idx = 0  # right
        elif state[0] > goal[0]:
            action_idx = 1  # left
        elif state[1] < goal[1]:
            action_idx = 2  # down
        elif state[1] > goal[1]:
            action_idx = 3  # up
        else:
            action_idx = 4  # stay
        
        action = np.zeros(self.action_space.n)
        action[action_idx] = 1
        return action


class KeyDoorVecEnv(BaseEnv):
    """Vectorized Key-Door environment for parallel execution."""

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self._keys = np.array([env.key for env in envs])
        self._doors = np.array([env.door for env in envs])
        self._dim = envs[0].dim
        self._horizon = envs[0].horizon
        self.markovian = envs[0].markovian
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

    @property
    def horizon(self):
        return self._horizon

    def sample_state(self):
        return np.random.randint(0, self._dim, (self._num_envs, 2)).astype(float)

    def sample_action(self):
        actions = np.zeros((self._num_envs, self.action_dim))
        actions[np.arange(self._num_envs), np.random.randint(0, self.action_dim, self._num_envs)] = 1
        return actions

    def sample_flags(self, states):
        """Sample random key/door possession flags for data collection."""
        have_keys = np.random.random(self._num_envs) > 0.5
        have_doors = have_keys & (np.random.random(self._num_envs) > 0.5)
        return have_keys, have_doors

    def reset(self):
        self.current_step = np.zeros(self._num_envs, dtype=int)
        self.states = np.zeros((self._num_envs, 2), dtype=float)
        self.have_keys = np.zeros(self._num_envs, dtype=bool)
        self.have_doors = np.zeros(self._num_envs, dtype=bool)
        return self.states.copy()

    def transit(self, states, actions, have_keys, have_doors):
        action_idxs = np.argmax(actions, axis=1)
        
        # Apply movements: 0=right, 1=left, 2=down, 3=up
        states[:, 0] += (action_idxs == 0).astype(float)
        states[:, 0] -= (action_idxs == 1).astype(float)
        states[:, 1] += (action_idxs == 2).astype(float)
        states[:, 1] -= (action_idxs == 3).astype(float)
        
        states = np.clip(states, 0, self._dim - 1)

        # Check key pickup
        at_keys = np.all(states == self._keys, axis=1) & ~have_keys
        have_keys = at_keys | have_keys
        
        # Check door reach (only counts if have key)
        at_doors = have_keys & np.all(states == self._doors, axis=1) & ~have_doors
        have_doors = at_doors | have_doors

        # Compute rewards
        if self.markovian:
            rewards = have_keys.astype(float) + have_doors.astype(float)
        else:
            rewards = at_keys.astype(float) + at_doors.astype(float)

        return states, rewards, have_keys, have_doors

    def step(self, actions):
        if np.any(self.current_step >= self._horizon):
            raise ValueError("Episode has already ended for some environments")

        self.states, rewards, self.have_keys, self.have_doors = self.transit(
            self.states, actions, self.have_keys, self.have_doors
        )
        self.current_step += 1
        dones = self.current_step >= self._horizon
        return self.states.copy(), rewards, dones, {}

    def opt_action(self, states, have_keys):
        actions = np.array([
            env.opt_action(state, have_key) 
            for env, state, have_key in zip(self._envs, states, have_keys)
        ])
        return actions
