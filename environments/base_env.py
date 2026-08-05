"""Base class for the gridworld and navigation environments."""

import gym
import numpy as np


class BaseEnv(gym.Env):
    """Base class for all environments."""

    def reset(self):
        raise NotImplementedError

    def transit(self, state, action):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode="human"):
        pass

    def deploy_eval(self, ctrl):
        return self.deploy(ctrl)

    def deploy(self, ctrl):
        """Run a controller in the environment and return trajectories."""
        ob = self.reset()
        obs, acts, next_obs, rews = [], [], [], []
        done = False

        while not done:
            act = ctrl.act(ob)
            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)
            rews.append(rew)
            next_obs.append(ob)
            done = np.any(done)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)

        return obs, acts, next_obs, rews
