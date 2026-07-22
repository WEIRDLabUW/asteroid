"""
Rollout policies for data collection.

Policies:
- ExpertPolicy: Uses environment's optimal action
- RandomPolicy: Uses random actions
- MLPPolicy: Uses trained MLP model
- TransformerPolicy: Uses trained Decision Transformer with context accumulation
- HybridPolicy: Mix of expert and learned policy
- ContextAccumulationPolicy: Transformer for earlier episodes, expert for last
"""

import numpy as np
import scipy.special
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePolicy:
    """Base class for rollout policies."""
    
    def __init__(self):
        self.env = None

    def set_env(self, env):
        """Set the environment for the policy."""
        self.env = env

    def reset(self):
        """Reset policy state (called at start of rollout)."""
        pass

    def get_action(self, states):
        """Get action for given states."""
        raise NotImplementedError

    def update_context(self, states, actions, rewards, dones):
        """Update policy context with new transition."""
        pass


class ExpertPolicy(BasePolicy):
    """Policy that returns the optimal action from the environment."""

    def get_action(self, states):
        if hasattr(self.env, "have_keys"):
            return self.env.opt_action(states, self.env.have_keys)
        return self.env.opt_action(states)


class RandomPolicy(BasePolicy):
    """Policy that returns random actions."""

    def get_action(self, states):
        return np.array([env.sample_action() for env in self.env._envs])


class NoisyExpertPolicy(BasePolicy):
    """
    Epsilon-greedy expert policy for AAWR data collection.
    
    With probability (1 - epsilon): use expert action
    With probability epsilon: use random action
    """

    def __init__(self, epsilon=0.25):
        """
        Args:
            epsilon: Probability of taking a random action (default: 0.25)
        """
        super().__init__()
        self.epsilon = epsilon

    def get_action(self, states):
        """Get action with epsilon-greedy exploration."""
        # Get expert actions
        if hasattr(self.env, "have_keys"):
            expert_actions = self.env.opt_action(states, self.env.have_keys)
        else:
            expert_actions = self.env.opt_action(states)
        
        # Get random actions
        random_actions = np.array([env.sample_action() for env in self.env._envs])
        
        # Epsilon-greedy selection
        batch_size = expert_actions.shape[0]
        use_random = np.random.rand(batch_size) < self.epsilon
        
        # Select expert or random based on epsilon
        actions = np.where(use_random[:, None], random_actions, expert_actions)
        return actions


class MLPPolicy(BasePolicy):
    """Policy using a trained MLP model."""

    def __init__(self, model, temp=1.0):
        super().__init__()
        self.model = model
        self.temp = temp

    @torch.no_grad()
    def get_action(self, states):
        states = np.array(states)
        x = torch.from_numpy(states).float().to(device)
        logits = self.model(x).cpu().numpy()
        probs = scipy.special.softmax(logits / self.temp, axis=1)
        batch_size, num_actions = probs.shape
        action_ids = np.array([
            np.random.choice(num_actions, p=probs[i]) for i in range(batch_size)
        ])
        actions = np.zeros((batch_size, num_actions))
        actions[np.arange(batch_size), action_ids] = 1.0
        return actions


class TransformerPolicy(BasePolicy):
    """
    Policy using a trained Decision Transformer.
    Accumulates context over time and uses it to predict actions.
    """

    def __init__(self, model, temp=0.1, context_horizon=None, env_horizon=None, 
                 sliding_window=True, use_value_guide=False):
        """
        Args:
            model: Trained Decision Transformer model
            temp: Temperature for action sampling (lower = more greedy)
            context_horizon: Maximum context length (defaults to model horizon)
            env_horizon: Steps per episode
            sliding_window: If True, use sliding window for context trimming
            use_value_guide: If True, use value-guided action selection
        """
        super().__init__()
        self.model = model
        self.temp = temp
        self.context_horizon = context_horizon or model.horizon
        self.env_horizon = env_horizon
        self.sliding_window = sliding_window
        self.use_value_guide = use_value_guide
        self.continuous_action = getattr(model, 'continuous_action', False)
        
        # Context buffers
        self.context_states = []
        self.context_actions = []
        self.context_rewards = []
        self.context_dones = []

    def reset(self):
        """Clear context buffers."""
        self.context_states = []
        self.context_actions = []
        self.context_rewards = []
        self.context_dones = []

    def _get_context_tensors(self):
        """Convert context lists to tensors, trimmed to context_horizon."""
        states = torch.from_numpy(np.stack(self.context_states, axis=1)).float().to(device)
        actions = torch.from_numpy(np.stack(self.context_actions, axis=1)).float().to(device)
        rewards = torch.from_numpy(np.stack(self.context_rewards, axis=1)).float().to(device)
        dones = torch.from_numpy(np.stack(self.context_dones, axis=1)).float().to(device)
        
        # Trim to context horizon if needed
        if states.shape[1] > self.model.horizon - 1:
            if self.sliding_window:
                # Simple sliding window
                states = states[:, -(self.context_horizon - 1):]
                actions = actions[:, -(self.context_horizon - 1):]
                rewards = rewards[:, -(self.context_horizon - 1):]
                dones = dones[:, -(self.context_horizon - 1):]
            else:
                # Trim to episode boundary
                trimmed_dones = dones[:, -self.context_horizon:]
                first_done_index = (
                    torch.argmax(trimmed_dones, dim=1) + 1 
                    + (states.shape[1] - self.context_horizon)
                )
                start_idx = first_done_index.min()
                states = states[:, start_idx:]
                actions = actions[:, start_idx:]
                rewards = rewards[:, start_idx:]
                dones = dones[:, start_idx:]
        
        return states, actions, rewards, dones

    @torch.no_grad()
    def get_action(self, states):
        """Get action using the transformer model."""
        self.model.eval()
        current_states = torch.from_numpy(states).float().to(device)
        
        if len(self.context_states) < 1:
            action_output = self.model.get_action(current_states, None, None, None, None)
        else:
            ctx_states, ctx_actions, ctx_rewards, ctx_dones = self._get_context_tensors()
            action_output = self.model.get_action(
                current_states, ctx_states, ctx_actions, ctx_rewards, ctx_dones
            )
        
        if self.continuous_action:
            # Handle continuous actions (distribution output)
            if self.temp < 1.0:
                action = action_output.mean
            else:
                action = action_output.sample()
            return action.cpu().numpy()
        else:
            # Handle discrete actions (logits output)
            logits = action_output.cpu().numpy()
            probs = scipy.special.softmax(logits / self.temp, axis=1)
            batch_size, num_actions = probs.shape
            action_ids = np.array([
                np.random.choice(num_actions, p=probs[i]) for i in range(batch_size)
            ])
            actions = np.zeros((batch_size, num_actions))
            actions[np.arange(batch_size), action_ids] = 1.0
            return actions

    def update_context(self, states, actions, rewards, dones):
        """Add new transition to context."""
        self.context_states.append(states)
        self.context_actions.append(actions)
        self.context_rewards.append(rewards)
        self.context_dones.append(dones)


class HybridPolicy(TransformerPolicy):
    """
    Mix of expert and learned policy.
    With probability beta, uses expert action; otherwise uses transformer.
    """

    def __init__(self, model, temp=0.1, context_horizon=None, env_horizon=None, 
                 sliding_window=True, beta=0.5):
        super().__init__(model, temp, context_horizon, env_horizon, sliding_window)
        self.expert = ExpertPolicy()
        self.beta = beta

    def set_env(self, env):
        super().set_env(env)
        self.expert.set_env(env)

    def get_action(self, states):
        base_action = super().get_action(states)
        expert_action = self.expert.get_action(states)
        batch_size = base_action.shape[0]
        use_expert = np.random.rand(batch_size) < self.beta
        return np.where(use_expert[:, None], expert_action, base_action)


class ContextAccumulationPolicy(TransformerPolicy):
    """
    Policy for context accumulation training (DAgger-style).
    Uses transformer for earlier episodes, expert for the last episode.
    """

    def __init__(self, model, temp=0.1, context_horizon=None, env_horizon=None, 
                 sliding_window=True, beta=0.0):
        super().__init__(model, temp, context_horizon, env_horizon, sliding_window)
        self.expert = ExpertPolicy()
        self.num_episodes = context_horizon // env_horizon if env_horizon else 1
        self.current_episode = 0

    def set_env(self, env):
        super().set_env(env)
        self.expert.set_env(env)

    def reset(self):
        super().reset()
        self.current_episode = 0

    def get_action(self, states):
        if self.current_episode >= self.num_episodes - 1:
            # Last episode: use expert
            return self.expert.get_action(states)
        else:
            # Earlier episodes: use transformer
            return super().get_action(states)

    def update_context(self, states, actions, rewards, dones):
        super().update_context(states, actions, rewards, dones)
        if np.any(dones):
            self.current_episode += 1


def get_rollout_policy(policy_type, model=None, temp=1.0, context_horizon=None, 
                       env_horizon=None, sliding_window=False, beta=0.0, 
                       use_value_guide=False, context_accumulation=False, epsilon=0.25):
    """
    Factory function to create rollout policies.
    
    Args:
        policy_type: "expert", "random", "noisy_expert", "mlp", "decision_transformer", or "hybrid"
        model: Trained model (required for learned policies)
        temp: Sampling temperature
        context_horizon: Maximum context length
        env_horizon: Steps per episode
        sliding_window: Use sliding window for context
        beta: Probability of using expert (for hybrid)
        use_value_guide: Use value-guided action selection
        context_accumulation: If True, use ContextAccumulationPolicy
        epsilon: Probability of random action (for noisy_expert)
    
    Returns:
        Policy instance
    """
    # Allow beta for noisy_expert, otherwise must be 0
    if policy_type != "noisy_expert":
        assert beta == 0.0, "Beta must be 0.0 for context accumulation"
    assert use_value_guide == False, "Value guide must be False for context accumulation"
    # assert temp == 1.0, "Temperature must be 1.0 for context accumulation"
    # if "spoc" not in policy_type:
    #     assert sliding_window == False, "Sliding window must be False for non-spoc policies"

    if context_accumulation:
        return ContextAccumulationPolicy(
            model, temp, context_horizon, env_horizon, sliding_window, beta
        )
    
    if policy_type == "expert":
        return ExpertPolicy()
    elif policy_type == "random":
        return RandomPolicy()
    elif policy_type == "noisy_expert":
        return NoisyExpertPolicy(epsilon=epsilon)
    elif policy_type == "mlp":
        if model is None:
            raise ValueError("Model required for MLP policy")
        return MLPPolicy(model, temp)
    elif policy_type == "decision_transformer":
        if model is None:
            raise ValueError("Model required for Transformer policy")
        return TransformerPolicy(
            model, temp, context_horizon, env_horizon, sliding_window, use_value_guide
        )
    elif policy_type == "hybrid":
        if model is None:
            raise ValueError("Model required for Hybrid policy")
        return HybridPolicy(
            model, temp, context_horizon, env_horizon, sliding_window, beta
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


class TransformerCNNPolicy():
    """
    Policy using a trained Decision Transformer.
    Accumulates context over time and uses it to predict actions.
    """

    def __init__(self, model, temp, context_horizon):
        """
        Args:
            model: Trained Decision Transformer model
            temp: Temperature for action sampling (lower = more greedy)
            context_horizon: Maximum context length (defaults to model horizon)
            env_horizon: Steps per episode
            sliding_window: If True, use sliding window for context trimming
            use_value_guide: If True, use value-guided action selection
            action_stats: Optional dict with 'mean' and 'std' for action denormalization
        """
        super().__init__()
        self.model = model
        self.temp = temp
        self.context_horizon = context_horizon
        
        # Context buffers
        self.context_states = None
        self.context_actions = None
        self.context_rewards = None
        self.context_dones = None

    def reset(self, resets):
        """Clear context buffers."""
        if self.context_states is None:
            self.context_states = [[] for _ in range(len(resets))]
            self.context_actions = [[] for _ in range(len(resets))]
            self.context_rewards = [[] for _ in range(len(resets))]
            self.context_dones = [[] for _ in range(len(resets))]
        else:
            for i, reset in enumerate(resets):
                if reset:
                    self.context_states[i] = []
                    self.context_actions[i] = []
                    self.context_rewards[i] = []
                    self.context_dones[i] = []

    def _get_context_tensors(self, current_states):
        # """Convert context lists to tensors, trimmed to context_horizon."""
        # states = torch.from_numpy(np.stack(self.context_states, axis=1)).float().to(device)
        # actions = torch.from_numpy(np.stack(self.context_actions, axis=1)).float().to(device)
        # rewards = torch.from_numpy(np.stack(self.context_rewards, axis=1)).float().to(device)
        # dones = torch.from_numpy(np.stack(self.context_dones, axis=1)).float().to(device)
        
        # # Trim to context horizon if needed
        # if states.shape[1] > self.model.horizon - 1:
        #     if self.sliding_window:
        #         # Simple sliding window
        #         states = states[:, -(self.context_horizon - 1):]
        #         actions = actions[:, -(self.context_horizon - 1):]
        #         rewards = rewards[:, -(self.context_horizon - 1):]
        #         dones = dones[:, -(self.context_horizon - 1):]
        #     else:
        #         # Trim to episode boundary
        #         trimmed_dones = dones[:, -self.context_horizon:]
        #         first_done_index = (
        #             torch.argmax(trimmed_dones, dim=1) + 1 
        #             + (states.shape[1] - self.context_horizon)
        #         )
        #         start_idx = first_done_index.min()
        #         states = states[:, start_idx:]
        #         actions = actions[:, start_idx:]
        #         rewards = rewards[:, start_idx:]
        #         dones = dones[:, start_idx:]
        
        # return states, actions, rewards, dones
        """Convert list to padded tensors"""
        batch_size = len(self.context_states)
        max_len = max(len(s) for s in self.context_states)+1  # +1 for current state
        states = torch.zeros((batch_size, max_len, *current_states[0].shape), dtype=torch.float32).to(device)
        actions = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)
        rewards = torch.zeros((batch_size, max_len), dtype=torch.float32).to(device)
        dones = torch.zeros((batch_size, max_len), dtype=torch.float32).to(device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.float32).to(device)
        for i in range(batch_size):
            seq_len = len(self.context_states[i])
            if seq_len > 0:
                states[i, :seq_len] = torch.from_numpy(np.stack(self.context_states[i], axis=0)).float().to(device)
                actions[i, 1:seq_len+1] = torch.from_numpy(np.stack(self.context_actions[i], axis=0)).long().to(device)
                rewards[i, 1:seq_len+1] = torch.from_numpy(np.array(self.context_rewards[i])).float().to(device)
                dones[i, 1:seq_len+1] = torch.from_numpy(np.array(self.context_dones[i])).float().to(device)
            states[i, seq_len] = torch.from_numpy(current_states[i]).float().to(device)  # Add current state as last in sequence
            attention_mask[i, :seq_len+1] = 1.0  # Mask for valid tokens
        return states, actions, rewards, dones, attention_mask
        

    @torch.no_grad()
    def get_action(self, states):
        """Get action using the transformer model."""
        self.model.eval()
        # current_states = torch.from_numpy(states).float().to(device)
        input_states, input_actions, input_rewards, input_dones, attention_mask = self._get_context_tensors(states)
        # print("Input states shape:", input_states.shape)
        # print("Input actions shape:", input_actions.shape)
        # print("Input rewards shape:", input_rewards.shape)
        # print("Input dones shape:", input_dones.shape)
        action_output = self.model.get_action(
            input_states, input_actions, input_rewards, input_dones, attention_mask
        )
        logits = action_output.cpu().numpy()
        probs = scipy.special.softmax(logits / self.temp, axis=1)
        batch_size, num_actions = probs.shape
        actions = np.array([
            np.random.choice(num_actions, p=probs[i]) for i in range(batch_size)
        ]) # Batch of action indices
        return actions

    def update_context(self, states, actions, rewards, dones):
        """Add new transition to context."""
        for i in range(len(states)):
            self.context_states[i].append(states[i])
            self.context_actions[i].append(actions[i])
            self.context_rewards[i].append(rewards[i])
