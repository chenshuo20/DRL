import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.context_len = context_len

        self.q_net = nn.Linear(hidden_dim, hidden_dim)
        self.k_net = nn.Linear(hidden_dim, hidden_dim)
        self.v_net = nn.Linear(hidden_dim, hidden_dim)

        self.proj_net = nn.Linear(hidden_dim, hidden_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((context_len, context_len))
        mask = torch.tril(ones).view(1, 1, context_len, context_len)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))

        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(hidden_dim, context_len, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_heads, n_blocks, hidden_dim, context_len, drop_p, action_space, state_mean, state_std, reward_scale, max_timestep, device):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # transformer blocks
        self.context_len = context_len
        blocks = [Block(hidden_dim, 3 * context_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_rtg = torch.nn.Linear(1, hidden_dim)
        self.embed_state = torch.nn.Linear(state_dim, hidden_dim)

        self.embed_action = torch.nn.Linear(action_dim, hidden_dim)

        # prediction heads
        self.predict_rtg = torch.nn.Linear(hidden_dim, 1)
        self.predict_state = torch.nn.Linear(hidden_dim, state_dim)
        self.predict_action = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32)
        self.reward_scale = reward_scale
        self.max_timestep = max_timestep
        self.to(device)

    def _norm_action(self, action):
        # linearly project actio to [action_space.low, action_space.high]
        ############################
        # YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################
    
    def _norm_state(self, state):
        # normalize state to zero mean and unit variance
        ############################
        # YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################
    
    def _norm_reward_to_go(self, reward_to_go):
        return reward_to_go / self.reward_scale

    def __repr__(self):
        return "DecisionTransformer"
    
    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        return super().to(device)

    def forward(self, states, actions, rewards_to_go, timesteps):
        states = self._norm_state(states)
        rewards_to_go = self._norm_reward_to_go(rewards_to_go)
        B, T, _ = states.shape


        ############################
        # time embeddings are added to each token in the sequence
        # YOUR IMPLEMENTATION HERE #
        time_embeddings = None
        state_embeddings = None
        action_embeddings = None
        rewards_to_go_embeddings = None

        ############################

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (rewards_to_go_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.blocks(h)
        
        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus the 3 input variables at that timestep.
        h = h.reshape(B, T, 3, self.hidden_dim).permute(0, 2, 1, 3)

        # get predictions
        ############################
        # YOUR IMPLEMENTATION HERE #
        # predict action given the decoded token of state, which contains information about previous transitions and current r, s
        action_preds = None 
        # (optional) predict state given the decoded token of action, which contains information about previous transitions and current r, s, a
        state_preds = None
        # (optional) predict rtg given the decoded token of action, which contains information about previous transitions and current r, s, a
        reward_to_go_preds = None
        ############################
        action_preds = self._norm_action(action_preds)
        return state_preds, action_preds, reward_to_go_preds
    
    def save(self, save_name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.state_dict(), os.path.join('models', f'{save_name}.pt'))
    
    def load(self, load_name):
        self.load_state_dict(torch.load(os.path.join('models', f'{load_name}.pt')))