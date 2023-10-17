from hydra.utils import instantiate
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation):
        super(QNetwork, self).__init__()
        self.q_head = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        Qs = self.q_head(state)
        return Qs


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            instantiate(activation),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        """
        Get the Q value of the current state and action using dueling network
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        feature = self.feature_layer(state)
        # print(self.value_head(feature).size(), self.advantage_head(feature).size())
        Qs = self.value_head(feature) + self.advantage_head(feature)
        ############################
        return Qs
