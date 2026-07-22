"""2D point-mass navigation to a hidden goal (single + vectorized)."""

import gym
import numpy as np

from environments.base_env import BaseEnv


class NavigationEnv(BaseEnv):
    """
    2D continuous navigation with discrete action space.
    
    Actions: 20 directions uniformly spaced around a circle + 1 no-op action.
    State: 2D position in [-radius, radius]^2.
    Goal: 2D position on a semi-circle.
    
    Modes:
    - Episodic: Resets to origin between episodes
    - Non-episodic (reset_free): Continues from last position between episodes
    """

    def __init__(self, radius, goal, horizon, reset_free=False, goal_tolerance=0.1):
        self.radius = radius
        self.goal = np.array(goal)
        self.horizon = horizon
        self.reset_free = reset_free
        self.goal_tolerance = goal_tolerance
        self.dt = 0.1  # Step size
        
        self.state_dim = 2
        self.action_dim = 21  # 20 directions + no-op
        
        self.observation_space = gym.spaces.Box(
            low=-self.radius, high=self.radius, shape=(self.state_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.action_dim)
        
        # Build action map: 20 directions uniformly around circle + no-op
        angles = np.linspace(0, 2 * np.pi, self.action_dim - 1, endpoint=False)
        self.action_map = np.array([
            [np.cos(angle), np.sin(angle)] for angle in angles
        ] + [[0.0, 0.0]])  # Last action is no-op
        
        self.state = np.array([0.0, 0.0])

    def sample_state(self):
        return np.random.uniform(-self.radius, self.radius, 2)

    def sample_action(self):
        action = np.zeros(self.action_space.n)
        action[np.random.randint(0, self.action_space.n)] = 1
        return action
        
    def reset(self):
        self.current_step = 0
        if not self.reset_free:
            self.state = np.array([0.0, 0.0])
        return self.state.copy()

    def transit(self, state, action):
        action_idx = np.argmax(action)
        direction = self.action_map[action_idx]

        next_state = state + direction * self.dt
        next_state = np.clip(next_state, -self.radius, self.radius)
        
        dist = np.linalg.norm(next_state - self.goal)
        reward = float(dist < self.goal_tolerance)
        
        return next_state, reward

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
        """Return optimal action towards goal."""
        diff = self.goal - state
        dist = np.linalg.norm(diff)
        
        if dist < self.goal_tolerance:
            # At goal, use no-op
            action = np.zeros(self.action_space.n)
            action[-1] = 1.0
            return action
        
        # Find closest action direction to goal direction
        direction = diff / dist
        dots = self.action_map @ direction
        action_idx = np.argmax(dots)
        
        action = np.zeros(self.action_space.n)
        action[action_idx] = 1
        return action


class NavigationVecEnv(BaseEnv):
    """Vectorized Navigation environment for parallel execution."""

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self._goals = np.array([env.goal for env in envs])
        self.radius = envs[0].radius
        self.horizon = envs[0].horizon
        self.dt = envs[0].dt
        self.goal_tolerance = envs[0].goal_tolerance
        self.reset_free = envs[0].reset_free
        self.action_map = envs[0].action_map
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        
        self.states = np.zeros((self._num_envs, 2), dtype=float)

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
        return np.random.uniform(-self.radius, self.radius, (self._num_envs, 2))
    
    def sample_action(self):
        actions = np.zeros((self._num_envs, self.action_dim))
        actions[np.arange(self._num_envs), np.random.randint(0, self.action_dim, self._num_envs)] = 1
        return actions
    
    def reset(self):
        self.current_step = np.zeros(self._num_envs, dtype=int)
        if not self.reset_free:
            self.states = np.zeros((self._num_envs, 2), dtype=float)
        return self.states.copy()

    def transit(self, states, actions):
        # Convert one-hot to directions
        if actions.shape[-1] == self.action_dim:
            action_idxs = np.argmax(actions, axis=1)
            directions = self.action_map[action_idxs]
        else:
            directions = actions
        
        next_states = states + directions * self.dt
        next_states = np.clip(next_states, -self.radius, self.radius)
        
        dists = np.linalg.norm(next_states - self._goals, axis=1)
        rewards = (dists < self.goal_tolerance).astype(float)
        
        return next_states, rewards

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
