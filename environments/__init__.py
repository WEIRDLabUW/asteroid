"""Environments, the environment factory, and rollout policies.

(``procgen_env`` is imported explicitly where needed since it requires the
optional ``procgen`` dependency.)
"""

from environments.create_envs import create_env
from environments.rollout_policy import get_rollout_policy
