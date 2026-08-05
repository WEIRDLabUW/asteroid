"""Model architectures: DPT Transformer, Decision Transformer (+ CNN variant), and the asymmetric critic."""

import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.distributions import TransformedDistribution, TanhTransform

def get_model(model_type, horizon, state_dim, action_dim, continuous_action, gmm_heads=1):
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.1
    shuffle = True
    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
        'continuous_action': continuous_action,
        'gmm_heads': gmm_heads,
    }
    if model_type == "decision_transformer":
        model = DecisionTransformer(config).to(device)
    elif model_type == "transformer":
        model = Transformer(config).to(device)
    elif model_type == "mlp":
        model = MLP(config).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_type}")
    return model

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_ctx=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=4,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.continuous_action = self.config['continuous_action']
        if self.continuous_action:
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
        else:
            self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])
            action_log_stds = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])
            if self.test:
                action_means = action_means[:, -1, :]
                action_log_stds = action_log_stds[:, -1, :]
            dist = torch.distributions.Normal(action_means, action_log_stds.exp())
            return dist
        else:
            preds = self.pred_actions(transformer_outputs['last_hidden_state'])

            if self.test:
                return preds[:, -1, :]
            return preds[:, 1:, :]


class ImageTransformer(Transformer):
    """Transformer class for image-based data."""

    def __init__(self, config):
        super().__init__(config)
        self.im_embd = 8

        size = self.config['image_size']
        size = (size - 3) // 2 + 1
        size = (size - 3) // 1 + 1

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        new_dim = self.im_embd + self.state_dim + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        query_images = x['query_images'][:, None, :]
        query_states = x['query_states'][:, None, :]
        context_images = x['context_images']
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        batch_size = query_states.shape[0]

        image_seq = torch.cat([query_images, context_images], dim=1)
        image_seq = image_seq.view(-1, *image_seq.size()[2:])

        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)

        context_states = torch.cat([query_states, context_states], dim=1)
        context_actions = torch.cat([
            torch.zeros(batch_size, 1, self.action_dim).to(device),
            context_actions,
        ], dim=1)
        context_rewards = torch.cat([
            torch.zeros(batch_size, 1, 1).to(device),
            context_rewards,
        ], dim=1)

        stacked_inputs = torch.cat([
            image_enc_seq,
            context_states,
            context_actions,
            context_rewards,
        ], dim=2)
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]


class MLP(nn.Module):
    """MLP class."""

    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config
        self.horizon = config['horizon']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        if self.config['continuous_action']:
            self.pred_action_means = nn.Linear(256, self.action_dim)
            self.pred_action_log_stds = nn.Linear(256, self.action_dim)
        else:
            self.pred_actions = nn.Linear(256, self.action_dim)
    
    def forward(self, x, sample_time=False):
        assert not sample_time, "MLP does not support sampling time."
        query_states = x
        query_states = query_states.view(-1, self.state_dim)
        query_states = query_states.to(device)
        if self.config['continuous_action']:
            action_means = self.pred_action_means(self.model(query_states))
            action_log_stds = self.pred_action_log_stds(self.model(query_states))
            dist = torch.distributions.Normal(action_means, action_log_stds.exp())
            return dist
        else:
            pred_actions = self.model(query_states)
            return pred_actions


class AsymmetricCritic(nn.Module):
    """
    Asymmetric Critic for AAWR (Asymmetric Advantage Weighted Regression).
    
    Uses privileged goal information during training to compute high-quality
    Q and V estimates. The policy only sees (state, reward, done) but the
    critic sees (state, reward, done, goal).
    
    Components:
    - Q-network: (state, action, goal) -> Q value
    - V-network: (state, goal) -> V value
    """

    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256, n_layers=3):
        """
        Args:
            state_dim: Dimension of state/observation
            action_dim: Dimension of action (for discrete, this is num_actions)
            goal_dim: Dimension of privileged goal information
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super(AsymmetricCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        
        # Q-network: (state, action, goal) -> Q value
        q_layers = []
        q_input_dim = state_dim + action_dim + goal_dim
        for i in range(n_layers):
            q_layers.append(nn.Linear(q_input_dim if i == 0 else hidden_dim, hidden_dim))
            q_layers.append(nn.ReLU())
        q_layers.append(nn.Linear(hidden_dim, 1))
        self.q_network = nn.Sequential(*q_layers)
        
        # V-network: (state, goal) -> V value
        v_layers = []
        v_input_dim = state_dim + goal_dim
        for i in range(n_layers):
            v_layers.append(nn.Linear(v_input_dim if i == 0 else hidden_dim, hidden_dim))
            v_layers.append(nn.ReLU())
        v_layers.append(nn.Linear(hidden_dim, 1))
        self.v_network = nn.Sequential(*v_layers)
        
        # Target Q-network for stable training
        self.q_target = None
        
    def init_target(self):
        """Initialize target Q-network as a copy of Q-network."""
        import copy
        self.q_target = copy.deepcopy(self.q_network)
        for param in self.q_target.parameters():
            param.requires_grad = False
            
    def update_target(self, tau=0.005):
        """Soft update target Q-network."""
        if self.q_target is None:
            self.init_target()
            return
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def q_value(self, state, action, goal):
        """
        Compute Q-value for (state, action, goal).
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim) - one-hot for discrete
            goal: (batch_size, goal_dim)
        
        Returns:
            q: (batch_size, 1)
        """
        x = torch.cat([state, action, goal], dim=-1)
        return self.q_network(x)
    
    def q_value_target(self, state, action, goal):
        """Compute Q-value using target network."""
        if self.q_target is None:
            return self.q_value(state, action, goal)
        x = torch.cat([state, action, goal], dim=-1)
        return self.q_target(x)
    
    def v_value(self, state, goal):
        """
        Compute V-value for (state, goal).
        
        Args:
            state: (batch_size, state_dim)
            goal: (batch_size, goal_dim)
        
        Returns:
            v: (batch_size, 1)
        """
        x = torch.cat([state, goal], dim=-1)
        return self.v_network(x)
    
    def advantage(self, state, action, goal):
        """
        Compute advantage A = Q(s,a,g) - V(s,g).
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            goal: (batch_size, goal_dim)
        
        Returns:
            advantage: (batch_size, 1)
        """
        q = self.q_value(state, action, goal)
        v = self.v_value(state, goal)
        return q - v


# class DecisionTransformer(nn.Module):
#     """Decision Transformer class."""

#     def __init__(self, config):
#         super(DecisionTransformer, self).__init__()

#         self.config = config
#         self.test = config['test']
#         self.horizon = self.config['horizon']
#         self.n_embd = self.config['n_embd']
#         self.n_layer = self.config['n_layer']
#         self.n_head = self.config['n_head']
#         self.state_dim = self.config['state_dim']
#         self.action_dim = self.config['action_dim']
#         self.dropout = self.config['dropout']

#         config = GPT2Config(
#             n_positions=4 * (1 + self.horizon),
#             n_embd=self.n_embd,
#             n_layer=self.n_layer,
#             n_head=1,
#             resid_pdrop=self.dropout,
#             embd_pdrop=self.dropout,
#             attn_pdrop=self.dropout,
#             use_cache=False,
#         )
#         self.transformer = GPT2Model(config)

#         self.embed_timestep = nn.Embedding(self.horizon, self.n_embd)
#         self.embed_transition = nn.Linear(
#             self.state_dim + self.action_dim + 2, self.n_embd)
#         self.continuous_action = self.config['continuous_action']
#         self.gmm_heads = self.config['gmm_heads']
#         if self.gmm_heads > 1:
#             assert self.continuous_action, "GMM only supported for continuous action spaces."
#             print("Using GMM with", self.gmm_heads, "heads")
#             # self.action_dim = self.action_dim * self.gmm_heads
#             self.pred_action_weights = nn.Linear(self.n_embd, self.gmm_heads)
#             self.pred_action_means = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
#             self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
#         elif self.continuous_action:
#             self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
#             self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
#         else:
#             self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        
#         self.content_mask = True

#     def forward(self, x, sample_time=False):
#         states = x['states']
#         actions = x['actions']
#         rewards = x['rewards']
#         dones = x['dones']
#         # print("Shapes:")
#         # print(f"States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}, Dones: {dones.shape}")
#         # print("Pad:", torch.zeros(states.shape[0], 1, self.action_dim).shape, actions[:, :-1, :].shape)
#         input_actions = torch.cat([
#             torch.zeros(states.shape[0], 1, self.action_dim).to(device),
#             actions[:, :-1, :],
#         ], dim=1)
#         input_rewards = torch.cat([
#             torch.zeros(states.shape[0], 1).to(device),
#             rewards[:, :-1],
#         ], dim=1)
#         input_dones = torch.cat([
#             torch.zeros(states.shape[0], 1).to(device),
#             dones[:, :-1],
#         ], dim=1)
#         # print(f"Shapes: states: {states.shape}, input_actions: {input_actions.shape}, rewards: {rewards.shape}, dones: {dones.shape}")
#         inputs = torch.cat([states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
#         # inputs = torch.cat([states, input_actions, rewards.unsqueeze(-1), dones.unsqueeze(-1)], dim=2)
#         timesteps = torch.arange(
#             inputs.shape[1], device=inputs.device).unsqueeze(0).expand(
#                 inputs.shape[0], -1)
#         if sample_time:
#             assert self.horizon > inputs.shape[1], "Horizon must be greater than the input sequence length."
#             initial_timestep = torch.randint(0, self.horizon - inputs.shape[1], (inputs.shape[0],), device=inputs.device)
#             timesteps = timesteps + initial_timestep.unsqueeze(1)

#         timestep_embeds = self.embed_timestep(timesteps)
#         inputs = self.embed_transition(inputs)
#         inputs = inputs + timestep_embeds

#         if self.content_mask:
#             attn_mask = self._build_split_context_mask(input_dones)
#         else:
#             attn_mask = None
#         transformer_outputs = self.transformer(inputs_embeds=inputs, attention_mask=attn_mask)
#         if self.gmm_heads > 1:
#             hidden = transformer_outputs['last_hidden_state']  # [B, T, H]
#             action_weights = self.pred_action_weights(hidden)
#             action_means = self.pred_action_means(hidden)
#             action_log_stds = self.pred_action_log_stds(hidden)
#             B, T, _ = action_means.shape
#             K = self.gmm_heads
#             D = self.action_dim
#             action_means    = action_means.reshape(B, T, K, D)
#             action_log_stds = action_log_stds.reshape(B, T, K, D)
#             LOG_SIG_MIN = -20.0
#             LOG_SIG_MAX =  20.0
#             action_log_stds = action_log_stds.clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#             action_stds = action_log_stds.exp()

#             mixture = torch.distributions.Categorical(logits=action_weights)
#             components = torch.distributions.Independent(
#                 torch.distributions.Normal(loc=action_means, scale=action_stds),  # batch: [B, T, K], event: D
#                 1
#             )
#             dist = torch.distributions.MixtureSameFamily(mixture, components)
#             return dist
#         if self.continuous_action:
#             action_means = self.pred_action_means(
#                 transformer_outputs['last_hidden_state'])
#             action_log_stds = self.pred_action_log_stds(
#                 transformer_outputs['last_hidden_state'])
#             dist = torch.distributions.Normal(action_means, action_log_stds.exp())
#             return dist
#         preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
#         return preds
    
#     def get_action(self, current_state, states, actions, rewards, dones):
#         # return self.debug_mlp(current_state)  # B x D -> B x A
#         if states is None: # current_state is B x D
#             input_states = current_state.unsqueeze(1)
#             input_actions = torch.zeros(current_state.shape[0], 1, self.action_dim).to(device)
#             input_rewards = torch.zeros(current_state.shape[0], 1).to(device)
#             input_dones = torch.zeros(current_state.shape[0], 1).to(device)
#         else: # states is B x T x D, actions is B x T x A, rewards is B x T, dones is B x T, current_state is B x D
#             input_states = torch.cat([
#                 states, current_state.unsqueeze(1)
#             ], dim=1) # B x (T+1) x D
#             input_actions = torch.cat([
#                 torch.zeros(states.shape[0], 1, self.action_dim).to(device),
#                 actions,
#             ], dim=1) # B x (T+1) x A
#             input_rewards = torch.cat([
#                 torch.zeros(states.shape[0], 1).to(device),
#                 rewards,
#             ], dim=1) # B x (T+1)
#             input_dones = torch.cat([
#                 torch.zeros(states.shape[0], 1).to(device),
#                 dones,
#             ], dim=1) # B x (T+1)

#             input_states = input_states[:, -self.horizon:, :]  # Keep only the last horizon states
#             input_actions = input_actions[:, -self.horizon:, :]
#             input_rewards = input_rewards[:, -self.horizon:]
#             input_dones = input_dones[:, -self.horizon:]
        
#         timesteps = torch.arange(
#             input_states.shape[1], device=input_states.device).unsqueeze(0).expand(
#                 input_states.shape[0], -1)
#         timestep_embeds = self.embed_timestep(timesteps)
#         # print(f"Shapes: input_states: {input_states.shape}, input_actions: {input_actions.shape}, rewards: {input_rewards.shape}, dones: {input_dones.shape}")
#         inputs = torch.cat([input_states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
#         inputs = self.embed_transition(inputs)
#         inputs = inputs + timestep_embeds
#         if self.content_mask:
#             attn_mask = self._build_split_context_mask(input_dones) # B x 1 x T x T
#             # print("Time:", inputs.shape[1])
#             # print("Attention mask:", attn_mask[0,0,:,:])
#             # print("------")
#             # breakpoint()
#         else:
#             attn_mask = None
#         transformer_outputs = self.transformer(inputs_embeds=inputs, attention_mask=attn_mask)
#         if self.gmm_heads > 1:
#             hidden = transformer_outputs['last_hidden_state']
#             action_weights = self.pred_action_weights(hidden)[:, -1, :]
#             action_means = self.pred_action_means(hidden)[:, -1, :]
#             action_log_stds = self.pred_action_log_stds(hidden)[:, -1, :]
#             B, _ = action_means.shape
#             K = self.gmm_heads
#             D = self.action_dim
#             action_means    = action_means.reshape(B, K, D)
#             action_log_stds = action_log_stds.reshape(B, K, D)
#             LOG_SIG_MIN = -20.0
#             LOG_SIG_MAX =  20.0
#             action_log_stds = action_log_stds.clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#             action_stds = action_log_stds.exp()
#             mixture = torch.distributions.Categorical(logits=action_weights)
#             components = torch.distributions.Independent(
#                 torch.distributions.Normal(loc=action_means, scale=action_stds),  # batch: [B, T, K], event: D
#                 1
#             )
#             dist = torch.distributions.MixtureSameFamily(mixture, components)
#             # return dist.sample()
#             return dist
#         if self.continuous_action:
#             action_means = self.pred_action_means(
#                 transformer_outputs['last_hidden_state'])[:, -1, :]
#             action_log_stds = self.pred_action_log_stds(
#                 transformer_outputs['last_hidden_state'])[:, -1, :]
#             dist = torch.distributions.Normal(action_means, action_log_stds.exp())
#             # return dist.sample()
#             return dist
#         preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x (T+1) x A
#         return preds[:, -1, :]  # Return the last action
    
#     def _build_split_context_mask(self, input_dones: torch.Tensor) -> torch.Tensor:
#         """
#         input_dones: [B, T] where value=1 indicates the step at that index is the end of a trajectory.
#                      (This matches how you constructed input_dones in forward/get_action: a left-shifted dones with leading zero.)
#         Returns: additive attention mask of shape [B, 1, T, T] with 0 for allowed, -inf for masked.
#         """
#         B, T = input_dones.shape
#         device = input_dones.device
#         done_bool = (input_dones > 0.5)

#         # last_done_idx_before[t]: index of the most recent 'done' strictly before t, else -1
#         last_done_idx = torch.full((B,), -1, device=device, dtype=torch.long)
#         last_done_before = torch.empty(B, T, device=device, dtype=torch.long)
#         for t in range(T):
#             last_done_before[:, t] = last_done_idx
#             # if this position itself is a done, it becomes the new "last done" for future steps
#             last_done_idx = torch.where(done_bool[:, t], torch.full_like(last_done_idx, t), last_done_idx)

#         idx = torch.arange(T, device=device).view(1, 1, 1, T)  # keys dimension
#         # allowed if key_index <= last_done_before[query]
#         allowed = (idx < last_done_before.view(B, 1, T, 1))  # [B,1,T,T] (query T x key T)

#         # We also respect causality implicitly: keys are always <= query index; masking stronger than needed is fine.
#         # Convert to additive mask: 0 for keep, -inf for block
#         attn_mask = torch.where(allowed, torch.zeros(1, device=device), torch.full((), float("-inf"), device=device))
#         return attn_mask  # [B,1,T,T]

class DecisionTransformer(nn.Module):
    """Decision Transformer class."""

    def __init__(self, config):
        super(DecisionTransformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        gpt_config = GPT2Config(
            n_positions=self.horizon,
            n_ctx=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt_config)

        self.embed_transition = nn.Linear(
            self.state_dim + self.action_dim + 2, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.continuous_action = self.config['continuous_action']
        self.gmm_heads = self.config['gmm_heads']
        if self.gmm_heads > 1:
            assert self.continuous_action, "GMM only supported for continuous action spaces."
            print("Using GMM with", self.gmm_heads, "heads")
            # self.action_dim = self.action_dim * self.gmm_heads
            self.pred_action_weights = nn.Linear(self.n_embd, self.gmm_heads)
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
        elif self.continuous_action:
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
        else:
            self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.action_proj = nn.Linear(self.action_dim, self.n_embd)
        self.tanh_action = config.get('tanh_action', False)
        self.low_noise_eval = config.get('low_noise_eval', False)
        
    def forward(self, x, query_actions=None, sample_time=False):
        states = x['states']
        actions = x['actions']
        rewards = x['rewards']
        dones = x['dones']
        input_actions = torch.cat([
            torch.zeros(states.shape[0], 1, self.action_dim).to(device),
            actions[:, :-1, :],
        ], dim=1)
        input_rewards = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            rewards[:, :-1],
        ], dim=1)
        input_dones = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            dones[:, :-1],
        ], dim=1)
        position_ids = None
        if sample_time:
            # assert False, "Sampling time not supported."
            position_ids = torch.arange(
                states.shape[1], device=states.device).unsqueeze(0).expand(
                    states.shape[0], -1)
            initial_timestep = torch.randint(0, self.horizon - states.shape[1], (states.shape[0],), device=states.device)
            position_ids = position_ids + initial_timestep.unsqueeze(1)

        inputs_ = torch.cat([states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        inputs = self.embed_transition(inputs_)
        inputs = self.embed_ln(inputs)

        transformer_outputs = self.transformer(inputs_embeds=inputs, position_ids=position_ids)

        # query_actions = x['actions'] if query_actions is None else query_actions
        # query_actions = query_actions.to(device)
        # query_actions = self.action_proj(query_actions)
        # value_pred_inputs = torch.cat([transformer_outputs['last_hidden_state'], query_actions], dim=-1)
        # value_preds = self.pred_values(value_pred_inputs) # B x T x 1
        value_preds = None

        if self.gmm_heads > 1:
            hidden = transformer_outputs['last_hidden_state']  # [B, T, H]
            action_weights = self.pred_action_weights(hidden)
            action_means = self.pred_action_means(hidden)
            action_log_stds = self.pred_action_log_stds(hidden)
            B, T, _ = action_means.shape
            K = self.gmm_heads
            D = self.action_dim
            action_means    = action_means.reshape(B, T, K, D)
            action_log_stds = action_log_stds.reshape(B, T, K, D)
            LOG_SIG_MIN = -20.0
            LOG_SIG_MAX =  20.0
            action_log_stds = action_log_stds.clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            action_stds = action_log_stds.exp()
            if not self.training and self.low_noise_eval:
                action_stds = torch.ones_like(action_stds) * 1e-4

            mixture = torch.distributions.Categorical(logits=action_weights)
            components = torch.distributions.Independent(
                torch.distributions.Normal(loc=action_means, scale=action_stds),  # batch: [B, T, K], event: D
                1
            )
            dist = torch.distributions.MixtureSameFamily(mixture, components)
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            return dist, value_preds
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])
            action_log_stds = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])
            action_stds = action_log_stds.exp()
            if not self.training and self.low_noise_eval:
                action_stds = torch.ones_like(action_stds) * 1e-4
            dist = torch.distributions.Normal(action_means, action_stds)
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            return dist, value_preds
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
        return preds, value_preds
    
    def get_action(self, current_state, states, actions, rewards, dones, return_transformer_outputs=False):
        # return_value = False
        # return self.debug_mlp(current_state)  # B x D -> B x A
        if states is None: # current_state is B x D
            input_states = current_state.unsqueeze(1)
            input_actions = torch.zeros(current_state.shape[0], 1, self.action_dim).to(device)
            input_rewards = torch.zeros(current_state.shape[0], 1).to(device)
            input_dones = torch.zeros(current_state.shape[0], 1).to(device)
        else: # states is B x T x D, actions is B x T x A, rewards is B x T, dones is B x T, current_state is B x D
            input_states = torch.cat([
                states, current_state.unsqueeze(1)
            ], dim=1) # B x (T+1) x D
            input_actions = torch.cat([
                torch.zeros(states.shape[0], 1, self.action_dim).to(device),
                actions,
            ], dim=1) # B x (T+1) x A
            input_rewards = torch.cat([
                torch.zeros(states.shape[0], 1).to(device),
                rewards,
            ], dim=1) # B x (T+1)
            input_dones = torch.cat([
                torch.zeros(states.shape[0], 1).to(device),
                dones,
            ], dim=1) # B x (T+1)

            input_states = input_states[:, -self.horizon:, :]  # Keep only the last horizon states
            input_actions = input_actions[:, -self.horizon:, :]
            input_rewards = input_rewards[:, -self.horizon:]
            input_dones = input_dones[:, -self.horizon:]
        
        inputs = torch.cat([input_states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        inputs = self.embed_transition(inputs)
        inputs = self.embed_ln(inputs)
        transformer_outputs = self.transformer(inputs_embeds=inputs)

        if self.gmm_heads > 1:
            hidden = transformer_outputs['last_hidden_state']
            action_weights = self.pred_action_weights(hidden)[:, -1, :]
            action_means = self.pred_action_means(hidden)[:, -1, :]
            action_log_stds = self.pred_action_log_stds(hidden)[:, -1, :]
            B, _ = action_means.shape
            K = self.gmm_heads
            D = self.action_dim
            action_means    = action_means.reshape(B, K, D)
            action_log_stds = action_log_stds.reshape(B, K, D)
            LOG_SIG_MIN = -20.0
            LOG_SIG_MAX =  20.0
            action_log_stds = action_log_stds.clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            action_stds = action_log_stds.exp()
            mixture = torch.distributions.Categorical(logits=action_weights)
            components = torch.distributions.Independent(
                torch.distributions.Normal(loc=action_means, scale=action_stds),  # batch: [B, T, K], event: D
                1
            )
            dist = torch.distributions.MixtureSameFamily(mixture, components)
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            if return_transformer_outputs:
                return dist, transformer_outputs['last_hidden_state'][:, -1, :]
            return dist
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])[:, -1, :]
            action_log_stds = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])[:, -1, :]
            dist = torch.distributions.Normal(action_means, action_log_stds.exp())
            if self.tanh_action:
                dist = TransformedDistribution(dist, TanhTransform())
            if return_transformer_outputs:
                return dist, transformer_outputs['last_hidden_state'][:, -1, :]
            return dist
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x (T+1) x A
        if return_transformer_outputs:
            return preds[:, -1, :], transformer_outputs['last_hidden_state'][:, -1, :]
        return preds[:, -1, :]  # Return the last action

# from decision_transformer.gym.decision_transformer.models.trajectory_gpt2 import GPT2Model
# from decision_transformer.gym.decision_transformer.models.model import TrajectoryModel

# class DecisionTransformer(TrajectoryModel):
#     """Decision Transformer class."""

#     def __init__(self, config):
#         # super(DecisionTransformer, self).__init__()

#         self.config = config
#         self.test = config['test']
#         self.horizon = self.config['horizon']
#         self.n_embd = self.config['n_embd']
#         self.n_layer = self.config['n_layer']
#         self.n_head = self.config['n_head']
#         self.state_dim = self.config['state_dim']
#         self.action_dim = self.config['action_dim']
#         self.dropout = self.config['dropout']

#         super().__init__(self.state_dim, self.action_dim, max_length=self.horizon)

#         self.hidden_size = self.n_embd
#         config = transformers.GPT2Config(
#             vocab_size=1,  # doesn't matter -- we don't use the vocab
#             n_embd=self.hidden_size,
#             n_layer=self.n_layer,
#             n_head=self.n_head,
#             n_inner=4*self.hidden_size,
#             n_positions=self.horizon,
#             resid_pdrop=self.dropout,
#             attn_pdrop=self.dropout
#         )
#         self.transformer = GPT2Model(config)

#         self.embed_timestep = nn.Embedding(self.horizon, self.hidden_size)
#         self.embed_return = torch.nn.Linear(1, self.hidden_size)
#         self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
#         self.embed_action = torch.nn.Linear(self.action_dim, self.hidden_size)

#         self.transformer = GPT2Model(config)
#         self.embed_ln = nn.LayerNorm(self.hidden_size)
#         self.action_tanh = True
#         self.predict_action = nn.Sequential(
#             *([nn.Linear(self.hidden_size, self.action_dim)] + ([nn.Tanh()] if self.action_tanh else []))
#         )
#         self.gmm_heads = self.config['gmm_heads']
#         if self.gmm_heads > 1:
#             assert self.continuous_action, "GMM only supported for continuous action spaces."
#             print("Using GMM with", self.gmm_heads, "heads")
#             # self.action_dim = self.action_dim * self.gmm_heads
#             self.pred_action_weights = nn.Linear(self.n_embd, self.gmm_heads)
#             self.pred_action_means = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
#             self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim * self.gmm_heads)
    
#     def forward(self, x, **kwargs):
#         states = x['states']
#         actions = x['actions']
#         rewards = x['rewards']
#         dones = x['dones']
#         batch_size, seq_length = states.shape[0], states.shape[1]
#         attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

#         input_actions = torch.cat([
#             torch.zeros(states.shape[0], 1, self.action_dim).to(device),
#             actions[:, :-1, :],
#         ], dim=1)
#         input_rewards = torch.cat([
#             torch.zeros(states.shape[0], 1).to(device),
#             rewards[:, :-1],
#         ], dim=1)
#         input_dones = torch.cat([
#             torch.zeros(states.shape[0], 1).to(device),
#             dones[:, :-1],
#         ], dim=1)

#         state_embeddings = self.embed_state(states)
#         action_embeddings = self.embed_action(input_actions)
#         reward_embeddings = self.embed_return(input_rewards.unsqueeze(-1))
#         done_embeddings = self.embed_return(input_dones.unsqueeze(-1))
#         timesteps = torch.arange(seq_length, device=states.device).unsqueeze(0).expand(batch_size, -1)
#         time_embeddings = self.embed_timestep(timesteps)

#         state_embeddings = state_embeddings + time_embeddings
#         action_embeddings = action_embeddings + time_embeddings
#         reward_embeddings = reward_embeddings + time_embeddings
#         done_embeddings = done_embeddings + time_embeddings

#         stacked_inputs = torch.stack(
#             (reward_embeddings, done_embeddings, state_embeddings, action_embeddings), dim=1
#         ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
#         stacked_inputs = self.embed_ln(stacked_inputs)

#         stacked_attention_mask = torch.stack(
#             (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
#         ).permute(0, 2, 1).reshape(batch_size, 4*seq_length)

#         transformer_outputs = self.transformer(
#             inputs_embeds=stacked_inputs,
#             attention_mask=stacked_attention_mask,
#         )
#         x = transformer_outputs['last_hidden_state']
#         x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

#         hidden = x[:, 2, :, :]  # (batch_size, seq_length, hidden_size)
#         if self.gmm_heads > 1:
#             action_weights = self.pred_action_weights(hidden)
#             action_means = self.pred_action_means(hidden)
#             action_log_stds = self.pred_action_log_stds(hidden)
#             B, T, _ = action_means.shape
#             K = self.gmm_heads
#             D = self.action_dim
#             action_means    = action_means.reshape(B, T, K, D)
#             action_log_stds = action_log_stds.reshape(B, T, K, D)
#             LOG_SIG_MIN = -20.0
#             LOG_SIG_MAX =  20.0
#             action_log_stds = action_log_stds.clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#             action_stds = action_log_stds.exp()

#             mixture = torch.distributions.Categorical(logits=action_weights)
#             components = torch.distributions.Independent(
#                 torch.distributions.Normal(loc=action_means, scale=action_stds),  # batch: [B, T, K], event: D
#                 1
#             )
#             dist = torch.distributions.MixtureSameFamily(mixture, components)
#             return dist, None
#         if self.continuous_action:
#             action_means = self.pred_action_means(
#                 transformer_outputs['last_hidden_state'])
#             action_log_stds = self.pred_action_log_stds(
#                 transformer_outputs['last_hidden_state'])
#             dist = torch.distributions.Normal(action_means, action_log_stds.exp())
#             return dist, None
#         preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
#         return preds, None
    
#     def get_action(self, current_state, states, actions, rewards, dones, return_transformer_outputs=False):
#         # return_value = False
#         # return self.debug_mlp(current_state)  # B x D -> B x A
#         if states is None: # current_state is B x D
#             input_states = current_state.unsqueeze(1)
#             input_actions = torch.zeros(current_state.shape[0], 1, self.action_dim).to(device)
#             input_rewards = torch.zeros(current_state.shape[0], 1).to(device)
#             input_dones = torch.zeros(current_state.shape[0], 1).to(device)
#         else: # states is B x T x D, actions is B x T x A, rewards is B x T, dones is B x T, current_state is B x D
#             input_states = torch.cat([
#                 states, current_state.unsqueeze(1)
#             ], dim=1) # B x (T+1) x D
#             input_actions = torch.cat([
#                 torch.zeros(states.shape[0], 1, self.action_dim).to(device),
#                 actions,
#             ], dim=1) # B x (T+1) x A
#             input_rewards = torch.cat([
#                 torch.zeros(states.shape[0], 1).to(device),
#                 rewards,
#             ], dim=1) # B x (T+1)
#             input_dones = torch.cat([
#                 torch.zeros(states.shape[0], 1).to(device),
#                 dones,
#             ], dim=1) # B x (T+1)

#             input_states = input_states[:, -self.horizon:, :]  # Keep only the last horizon states
#             input_actions = input_actions[:, -self.horizon:, :]
#             input_rewards = input_rewards[:, -self.horizon:]
#             input_dones = input_dones[:, -self.horizon:]
        
#         # timesteps = torch.arange(
#         #     input_states.shape[1], device=input_states.device).unsqueeze(0).expand(
#         #         input_states.shape[0], -1)
#         # timestep_embeds = self.embed_timestep(timesteps)
#         # print(f"Shapes: input_states: {input_states.shape}, input_actions: {input_actions.shape}, rewards: {input_rewards.shape}, dones: {input_dones.shape}")
#         inputs = torch.cat([input_states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
#         inputs = self.embed_transition(inputs)
#         inputs = self.embed_ln(inputs)
#         # inputs = self.get_input_embeddings(input_states, input_actions, input_rewards, input_dones)
#         # value_inputs = self.embed_value_transition(inputs)
#         # inputs = inputs + timestep_embeds
#         transformer_outputs = self.transformer(inputs_embeds=inputs)
#         # value_transformer_outputs = self.value_transformer(inputs_embeds=value_inputs)
#         # pred_values = self.pred_values(transformer_outputs['last_hidden_state'])[:, -1, :] # B x value_bins

#         if self.gmm_heads > 1:
#             hidden = transformer_outputs['last_hidden_state']
#             action_weights = self.pred_action_weights(hidden)[:, -1, :]
#             action_means = self.pred_action_means(hidden)[:, -1, :]
#             action_log_stds = self.pred_action_log_stds(hidden)[:, -1, :]
#             B, _ = action_means.shape
#             K = self.gmm_heads
#             D = self.action_dim
#             action_means    = action_means.reshape(B, K, D)
#             action_log_stds = action_log_stds.reshape(B, K, D)
#             LOG_SIG_MIN = -20.0
#             LOG_SIG_MAX =  20.0
#             action_log_stds = action_log_stds.clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#             action_stds = action_log_stds.exp()
#             mixture = torch.distributions.Categorical(logits=action_weights)
#             components = torch.distributions.Independent(
#                 torch.distributions.Normal(loc=action_means, scale=action_stds),  # batch: [B, T, K], event: D
#                 1
#             )
#             dist = torch.distributions.MixtureSameFamily(mixture, components)
#             # return dist.sample()
#             # if return_value:
#             #     return dist, pred_values
#             if return_transformer_outputs:
#                 return dist, transformer_outputs['last_hidden_state'][:, -1, :]
#             return dist
#         if self.continuous_action:
#             action_means = self.pred_action_means(
#                 transformer_outputs['last_hidden_state'])[:, -1, :]
#             action_log_stds = self.pred_action_log_stds(
#                 transformer_outputs['last_hidden_state'])[:, -1, :]
#             dist = torch.distributions.Normal(action_means, action_log_stds.exp())
#             # return dist.sample()
#             # if return_value:
#             #     return dist, pred_values
#             if return_transformer_outputs:
#                 return dist, transformer_outputs['last_hidden_state'][:, -1, :]
#             return dist
#         preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x (T+1) x A
#         # if return_value:
#         #     return preds[:, -1, :], pred_values
#         if return_transformer_outputs:
#             return preds[:, -1, :], transformer_outputs['last_hidden_state'][:, -1, :]
#         return preds[:, -1, :]  # Return the last action

# class DecisionTransformer(nn.Module):
#     """Decision Transformer class."""

#     def __init__(self, config):
#         super(DecisionTransformer, self).__init__()

#         self.config = config
#         self.test = config['test']
#         self.horizon = self.config['horizon']
#         self.n_embd = self.config['n_embd']
#         self.n_layer = self.config['n_layer']
#         self.n_head = self.config['n_head']
#         self.state_dim = self.config['state_dim']
#         self.action_dim = self.config['action_dim']
#         self.dropout = self.config['dropout']
#         self.continuous_action = self.config['continuous_action']

#         gpt_config = GPT2Config(
#             n_positions=self.horizon,
#             n_embd=self.n_embd,
#             n_layer=self.n_layer,
#             n_head=self.n_head,
#             resid_pdrop=self.dropout,
#             embd_pdrop=self.dropout,
#             attn_pdrop=self.dropout,
#             use_cache=False,
#         )
#         self.transformer = GPT2Model(gpt_config)

#         self.embed_state = nn.Linear(self.state_dim, self.n_embd)
#         self.embed_reward = nn.Embedding(2, self.n_embd)
#         self.embed_ln = nn.LayerNorm(self.n_embd)
        
#         self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
#         self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
#         self.action_proj = nn.Linear(self.action_dim, self.n_embd)
#         self.tanh_action = config['tanh_action']
#         self.low_noise_eval = config['low_noise_eval']
        
#     def forward(self, x, query_actions=None, sample_time=False):
#         states = x['states']
#         rewards = x['rewards']
#         dones = x['dones']
#         input_rewards = torch.cat([
#             torch.zeros(states.shape[0], 1).to(device),
#             rewards[:, :-1],
#         ], dim=1)
#         input_dones = torch.cat([
#             torch.zeros(states.shape[0], 1).to(device),
#             dones,
#         ], dim=1)
        

#         state_embeddings = self.embed_state(states)
#         reward_embeddings = self.embed_reward((input_rewards > 0).long())
#         position_ids = torch.arange(states.shape[1], device=states.device).unsqueeze(0).expand(states.shape[0], -1)
#         attention_mask = build_attn_mask_reward_state(input_dones)

#         batch_size, seq_length, _ = states.shape
#         stacked_inputs = torch.stack(
#             (reward_embeddings, state_embeddings), dim=1
#         ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.n_embd)
#         stacked_inputs = self.embed_ln(stacked_inputs)

#         stacked_position_ids = torch.stack(
#             (position_ids, position_ids), dim=1
#         ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

#         transformer_outputs = self.transformer(
#             inputs_embeds=stacked_inputs,
#             position_ids=stacked_position_ids,
#             attention_mask=attention_mask
#         )
#         last_hidden_state = transformer_outputs['last_hidden_state'].reshape(batch_size, seq_length, 2, self.n_embd)[:, :, 1, :]
#         action_means = self.pred_action_means(
#             last_hidden_state)
#         action_log_stds = self.pred_action_log_stds(
#             last_hidden_state)
#         action_stds = action_log_stds.exp()
#         dist = torch.distributions.Normal(action_means, action_stds)
#         if self.tanh_action:
#             dist = TransformedDistribution(dist, TanhTransform())
#         return dist, None
    
#     def get_action(self, current_state, states, actions, rewards, dones, return_transformer_outputs=False):
#         # return_value = False
#         # return self.debug_mlp(current_state)  # B x D -> B x A
#         if states is None: # current_state is B x D
#             input_states = current_state.unsqueeze(1)
#             input_rewards = torch.zeros(current_state.shape[0], 1).to(device)
#             # HACK
#             input_dones = torch.zeros(current_state.shape[0], 2).to(device)
#         else: # states is B x T x D, actions is B x T x A, rewards is B x T, dones is B x T, current_state is B x D
#             input_states = torch.cat([
#                 states, current_state.unsqueeze(1)
#             ], dim=1) # B x (T+1) x D
#             input_rewards = torch.cat([
#                 torch.zeros(states.shape[0], 1).to(device),
#                 rewards,
#             ], dim=1) # B x (T+1)
#             # HACK
#             input_dones = torch.cat([
#                 torch.zeros(states.shape[0], 1).to(device),
#                 dones,
#                 torch.zeros(states.shape[0], 1).to(device),
#             ], dim=1) # B x (T+1)

#             input_states = input_states[:, -self.horizon:, :]  # Keep only the last horizon states
#             # input_actions = input_actions[:, -self.horizon:, :]
#             input_rewards = input_rewards[:, -self.horizon:]
#             input_dones = input_dones[:, -self.horizon:]
        

#         state_embeddings = self.embed_state(input_states)
#         reward_embeddings = self.embed_reward((input_rewards > 0).long())
#         position_ids = torch.arange(input_states.shape[1], device=input_states.device).unsqueeze(0).expand(input_states.shape[0], -1)
#         batch_size, seq_length, _ = input_states.shape
#         stacked_inputs = torch.stack(
#             (reward_embeddings, state_embeddings), dim=1
#         ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.n_embd)
#         stacked_inputs = self.embed_ln(stacked_inputs)
#         stacked_position_ids = torch.stack(
#             (position_ids, position_ids), dim=1
#         ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)
#         attention_mask = build_attn_mask_reward_state(input_dones)
#         transformer_outputs = self.transformer(inputs_embeds=stacked_inputs, position_ids=stacked_position_ids, attention_mask=attention_mask)
#         last_hidden_state = transformer_outputs['last_hidden_state'].reshape(batch_size, seq_length, 2, self.n_embd)[:, :, 1, :]
#         action_means = self.pred_action_means(
#             last_hidden_state)[:, -1, :]
#         action_log_stds = self.pred_action_log_stds(
#             last_hidden_state)[:, -1, :]
#         dist = torch.distributions.Normal(action_means, action_log_stds.exp())
#         if self.tanh_action:
#             dist = TransformedDistribution(dist, TanhTransform())
#         if return_transformer_outputs:
#             return dist, last_hidden_state
#         return dist

# def build_attn_mask_reward_state(dones_padded: torch.Tensor) -> torch.Tensor:
#     """
#     dones_padded: [B, T+1], leading zero; tokens are [r0,s0,r1,s1,...,r_{T-1},s_{T-1}]
#     Returns additive mask [B,1,2T,2T] with 0 allowed and -inf masked.
#     """
#     B, Tp1 = dones_padded.shape
#     T = Tp1 - 1
#     device = dones_padded.device
#     TT = 2 * T

#     # Subtrajectory ids at each time t (0..T-1): increment when previous step ended
#     sub_id = torch.cumsum((dones_padded[:, :T] > 0.5).long(), dim=1)   # [B,T]
#     done_at_t = (dones_padded[:, 1:] > 0.5)                            # [B,T], true where time t is a done-step

#     t = torch.arange(T, device=device)
#     r_idx = 2 * t               # reward token columns/rows
#     s_idx = 2 * t + 1           # state  token columns/rows

#     allow = torch.zeros(B, TT, TT, dtype=torch.bool, device=device)

#     # (2) Rewards: attend only to self
#     allow[:, r_idx, r_idx] = True

#     # (3a) States: attend to states in same subtrajectory up to itself (causal in time)
#     same_sub = (sub_id[:, :, None] == sub_id[:, None, :])   # [B,T,T]
#     causal_t = (t[None, :, None] >= t[None, None, :])       # [1,T,T]
#     ss_mask  = same_sub & causal_t                          # [B,T,T]
#     allow[:, s_idx[:, None], s_idx[None, :]] = ss_mask

#     # (3b) States: also attend to {reward,state} at done-steps of earlier subtrajectories
#     earlier_done = done_at_t[:, None, :] & (sub_id[:, None, :] < sub_id[:, :, None])  # [B,T,T] (query time i, key time j)
#     allow[:, s_idx[:, None], (2 * t)[None, :]]   |= earlier_done  # to reward@done(j)
#     allow[:, s_idx[:, None], (2 * t + 1)[None, :]] |= earlier_done  # to state@done(j)

#     # (1) Global causal at token level
#     causal_tok = torch.tril(torch.ones(TT, TT, dtype=torch.bool, device=device))
#     allow &= causal_tok

#     # Convert to additive mask
#     mask = torch.where(allow, torch.zeros((), device=device), torch.full((), float('-inf'), device=device))
#     return mask.unsqueeze(1)


# import torch
# import torch.nn as nn
# from torch.distributions import Normal, Categorical, Independent, MixtureSameFamily, TransformedDistribution
# from torch.distributions.transforms import TanhTransform

# class DecisionTransformer(nn.Module):
#     """MLP variant using prev-done context; Gaussian or GMM head via config['gmm_heads']."""

#     def __init__(self, config):
#         super(DecisionTransformer, self).__init__()
#         self.config = config
#         self.test = config['test']
#         self.horizon = config['horizon']
#         self.n_embd = config['n_embd']
#         self.n_layer = config['n_layer']
#         self.n_head = config['n_head']
#         self.state_dim = config['state_dim']
#         self.action_dim = config['action_dim']
#         self.dropout = config['dropout']
#         self.continuous_action = config['continuous_action']
#         self.tanh_action = config['tanh_action']
#         self.low_noise_eval = config['low_noise_eval']
#         self.gmm_heads = int(config.get('gmm_heads', 1))

#         # Embeddings
#         self.embed_state  = nn.Linear(self.state_dim, self.n_embd)
#         self.embed_reward = nn.Embedding(2, self.n_embd)  # sign(r) ∈ {0,1}
#         self.embed_ln     = nn.LayerNorm(self.n_embd)

#         # Fuse [cur_s, prev_done_s, prev_done_r] -> h
#         self.fuse = nn.Sequential(
#             nn.Linear(3 * self.n_embd, self.n_embd),
#             nn.GELU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(self.n_embd, self.n_embd),
#             nn.GELU(),
#         )

#         # Heads
#         if self.gmm_heads > 1:
#             K = self.gmm_heads
#             self.pred_action_weights  = nn.Linear(self.n_embd, K)
#             self.pred_action_means    = nn.Linear(self.n_embd, K * self.action_dim)
#             self.pred_action_log_stds = nn.Linear(self.n_embd, K * self.action_dim)
#         else:
#             self.pred_action_means    = nn.Linear(self.n_embd, self.action_dim)
#             self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)

#     @staticmethod
#     def _prev_done_index(dones_padded: torch.Tensor):
#         B, Tp1 = dones_padded.shape
#         T = Tp1 - 1
#         done_at_t = (dones_padded[:, 1:] > 0.5)   # [B,T]
#         j = torch.arange(T, device=dones_padded.device) + 1
#         marks = done_at_t * j[None, :]
#         prev_plus, _ = torch.cummax(marks, dim=1)  # 0..T
#         return prev_plus - 1                        # -1 means none

#     @staticmethod
#     def _gather_time(x: torch.Tensor, t_idx: torch.Tensor):
#         B, T = t_idx.shape
#         device = x.device
#         valid = (t_idx >= 0)
#         safe_idx = t_idx.clamp_min(0)
#         b_idx = torch.arange(B, device=device)[:, None].expand(B, T)
#         gathered = x[b_idx, safe_idx]
#         return gathered * valid[..., None].to(x.dtype)

#     def _make_dist_from_h(self, h: torch.Tensor, last_only: bool = False):
#         """
#         h: [B,T,E] or [B,E]; returns torch.distributions.Distribution (Gaussian or GMM).
#         If last_only=True, treat h as [B,E] (single step).
#         """
#         LOG_SIG_MIN, LOG_SIG_MAX = -20.0, 20.0
#         if self.gmm_heads > 1:
#             K, A = self.gmm_heads, self.action_dim
#             if last_only:
#                 w   = self.pred_action_weights(h)                  # [B,K]
#                 mu  = self.pred_action_means(h).view(-1, K, A)     # [B,K,A]
#                 ls  = self.pred_action_log_stds(h).view(-1, K, A).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
#             else:
#                 B, T, _ = h.shape
#                 w   = self.pred_action_weights(h)                  # [B,T,K]
#                 mu  = self.pred_action_means(h).view(B, T, K, A)   # [B,T,K,A]
#                 ls  = self.pred_action_log_stds(h).view(B, T, K, A).clamp(LOG_SIG_MIN, LOG_SIG_MAX)

#             ls = ls.exp()
#             if self.low_noise_eval and not self.training:
#                 ls = torch.ones_like(ls) * 1e-2
#             comp = Independent(Normal(mu, ls), 1)            # event_dim=1 over A
#             mix  = Categorical(logits=w)
#             dist = MixtureSameFamily(mix, comp)
#         else:
#             if last_only:
#                 mu = self.pred_action_means(h)                     # [B,A]
#                 ls = self.pred_action_log_stds(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
#                 dist = Normal(mu, ls.exp())
#             else:
#                 mu = self.pred_action_means(h)                     # [B,T,A]
#                 ls = self.pred_action_log_stds(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
#                 dist = Normal(mu, ls.exp())

#         if self.tanh_action:
#             dist = TransformedDistribution(dist, TanhTransform())
#         return dist

#     def forward(self, x, query_actions=None, sample_time=False):
#         """
#         x: dict with 'states':[B,T,Ds], 'rewards':[B,T], 'dones':[B,T]
#         Uses prev-done (state,reward) + current state to predict per-step action dist.
#         """
#         states, rewards, dones = x['states'], x['rewards'], x['dones']
#         B, T, Ds = states.shape
#         device = states.device

#         dones_padded = torch.cat([torch.zeros(B, 1, device=device), dones], dim=1)  # [B,T+1]
#         prev_idx = self._prev_done_index(dones_padded)                               # [B,T]

#         prev_done_states  = self._gather_time(states, prev_idx)                      # [B,T,Ds]
#         prev_done_rewards = self._gather_time(rewards.unsqueeze(-1), prev_idx).squeeze(-1)  # [B,T]

#         cur_s_emb  = self.embed_ln(self.embed_state(states))
#         prev_s_emb = self.embed_ln(self.embed_state(prev_done_states))
#         prev_r_emb = self.embed_ln(self.embed_reward((prev_done_rewards > 0).long()))
#         h = self.fuse(torch.cat([cur_s_emb, prev_s_emb, prev_r_emb], dim=-1))        # [B,T,E]

#         dist = self._make_dist_from_h(h, last_only=False)
#         return dist, None

#     def get_action(self, current_state, states, actions, rewards, dones, return_transformer_outputs=False):
#         """
#         Returns a distribution over next action given history + current_state.
#         current_state: [B,Ds]
#         states: [B,T,Ds] or None; rewards,dones: [B,T] if states is not None.
#         """
#         device = current_state.device
#         if states is None:
#             prev_done_state  = torch.zeros_like(current_state)
#             prev_done_reward = torch.zeros(current_state.size(0), device=device)
#         else:
#             B, T, Ds = states.shape
#             states_c  = torch.cat([states, current_state.unsqueeze(1)], dim=1)[:, -self.horizon:, :]
#             rewards_c = torch.cat([torch.zeros(B, 1, device=device), rewards], dim=1)[:, -self.horizon:]
#             dones_c   = torch.cat([torch.zeros(B, 1, device=device), dones, torch.zeros(B, 1, device=device)], dim=1)[:, -self.horizon:]
#             Th = states_c.shape[1]
#             prev_idx = self._prev_done_index(dones_c)                                   # [B,Th]
#             last_idx = prev_idx[:, -1].unsqueeze(1)
#             prev_done_state  = self._gather_time(states_c,  last_idx).squeeze(1)        # [B,Ds]
#             prev_done_reward = self._gather_time(rewards_c.unsqueeze(-1), last_idx).squeeze(1).squeeze(-1)  # [B]

#         cur_s_emb  = self.embed_ln(self.embed_state(current_state))                     # [B,E]
#         prev_s_emb = self.embed_ln(self.embed_state(prev_done_state))                   # [B,E]
#         prev_r_emb = self.embed_ln(self.embed_reward((prev_done_reward > 0).long()))    # [B,E]
#         h = self.fuse(torch.cat([cur_s_emb, prev_s_emb, prev_r_emb], dim=-1))           # [B,E]

#         dist = self._make_dist_from_h(h, last_only=True)
#         if return_transformer_outputs:
#             return dist, h
#         return dist

class DecisionTransformerCnn(nn.Module):
    """Decision Transformer with image/grid encoder for observations.

    Observations arrive as (H, W, C) uint8/float from procgen / gymnasium.

    Encoder selection (automatic, based on spatial size):
    * **H ≥ 32** (e.g. 64×64 RGB): Atari-style CNN
      Conv(k=8,s=4) → Conv(k=4,s=2) → Conv(k=3,s=1) → Linear.
    * **H < 32** (e.g. 7×7 binary grid): lightweight MLP
      Flatten → Linear → ReLU → Linear → ReLU.

    Config must contain:
        obs: tuple (H, W, C) – raw observation shape
        n_embd: int – hidden dimension (encoder output dim)
        ... (all keys required by DecisionTransformer)
    """

    def __init__(self, config):
        super().__init__()
        gpt2_config = GPT2Config(
            n_positions=config['horizon'],
            n_ctx=config['horizon'],
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            resid_pdrop=config['dropout'],
            embd_pdrop=config['dropout'],
            attn_pdrop=config['dropout'],
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt2_config)
        obs_shape = config['obs']  # (H, W, C), (15x15x3)
        self.cnn_encoder = nn.Sequential(
            # nn.Conv2d(obs_shape[2], 64, kernel_size=3, stride=1),  # (H/2, W/2, 64)
            # nn.ReLU(),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1),  # (H/4, W/4, 32)
            # nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(obs_shape[0]//4 * obs_shape[1]//4 * 32, config['n_embd']),  # (n_embd)
            # nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
            dummy_output = self.cnn_encoder(dummy_input)
            state_dim = dummy_output.shape[1]
            self.obs_proj = nn.Linear(state_dim, config['n_embd'])
        self.cnn_encoder = nn.Sequential(
            self.cnn_encoder,
            self.obs_proj,
            nn.ReLU(),
        )
        self.action_embeds = nn.Embedding(config['action_dim'], config['n_embd'])

        # state_dim = np.prod(obs_shape)
        state_dim = config['n_embd']
        action_dim = config['action_dim']
        n_embd = config['n_embd']
        self.embed_transition = nn.Linear(
            2*config['n_embd'], n_embd)
        self.embed_ln = nn.LayerNorm(n_embd)
        self.pred_actions = nn.Linear(n_embd, action_dim)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_embd = n_embd

    def forward(self, x, **kwargs):
        B,T = x['states'].shape[0], x['states'].shape[1]
        # states = x['states'].view(B, T, self.state_dim)
        states = x['states'].view(B * T, *x['states'].shape[2:])  # (B*T, H, W, C)
        # breakpoint()
        states = states.permute(0, 3, 1, 2)
        states = self.cnn_encoder(states)  # (B*T, n_embd)
        states = states.view(B, T, self.n_embd)  # (B, T, n_embd)
        
        actions = x['actions']
        input_actions = torch.cat([
            torch.zeros((actions.shape[0], 1), dtype=torch.long).to(device),
            actions[:, :-1],
        ], dim=1)
        # breakpoint()
        # input_actions = input_actions.argmax(dim=-1)
        actions = self.action_embeds(input_actions)
        # rewards = x['rewards']
        # dones = x['dones']
        # input_actions = torch.cat([
        #     torch.zeros(states.shape[0], 1, self.action_dim).to(device),
        #     actions[:, :-1, :],
        # ], dim=1)
        # input_rewards = torch.cat([
        #     torch.zeros(states.shape[0], 1).to(device),
        #     rewards[:, :-1],
        # ], dim=1)
        # input_dones = torch.cat([
        #     torch.zeros(states.shape[0], 1).to(device),
        #     dones[:, :-1],
        # ], dim=1)
        # inputs_ = torch.cat([states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        inputs_ = torch.cat([states, actions], dim=2)
        inputs = self.embed_transition(inputs_)
        inputs = self.embed_ln(inputs)

        transformer_outputs = self.transformer(inputs_embeds=inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
        return preds

    def get_action(self, states, actions, rewards, dones, attention_mask):
        # current_state: [B, H, W, C] -> [B, hidden_dim]
        B,T = states.shape[0], states.shape[1]
        # states = states.view(B, T, self.state_dim)
        states = states.view(B * T, *states.shape[2:])  # (B*T, H, W, C)
        states = states.permute(0, 3, 1, 2)
        states = self.cnn_encoder(states)  # (B*T, n_embd)
        states = states.view(B, T, self.n_embd)  # (B, T, n_embd)
        actions = self.action_embeds(actions)
        # inputs_ = torch.cat([states, actions, rewards.unsqueeze(-1), dones.unsqueeze(-1)], dim=2)
        inputs_ = torch.cat([states, actions], dim=2)
        inputs = self.embed_transition(inputs_)
        inputs = self.embed_ln(inputs)
        transformer_outputs = self.transformer(inputs_embeds=inputs, attention_mask=attention_mask)
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
        seq_len = attention_mask.sum(dim=1).long()  # [B]
        last_preds = preds[torch.arange(B), seq_len - 1]  # [B, A]
        return last_preds

