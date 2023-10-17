import os
import torch
import torch.optim as optim
from copy import deepcopy
from model import QNetwork, DuelingQNetwork
from gymnasium.wrappers import TimeLimit

class DQNAgent:
    def __init__(self, state_size, action_size, cfg, device='cuda'):
        self.device = device
        self.use_double = cfg.use_double
        self.use_dueling = cfg.use_dueling
        self.target_update_interval = cfg.target_update_interval
        q_model = DuelingQNetwork if self.use_dueling else QNetwork

        self.q_net = q_model(state_size, action_size, cfg.hidden_size, cfg.activation).to(self.device)
        self.target_net = deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=cfg.lr)

        self.tau = cfg.tau
        # update the gamma we use in the Bellman equation for n-step DQN
        self.gamma = cfg.gamma ** cfg.nstep

    def soft_update(self, target, source):
        """
        Soft update the target network using the source network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    @torch.no_grad()
    def get_action(self, state):
        """
        Get the action according to the current state and Q value
        """

        ############################
        # YOUR IMPLEMENTATION HERE #
        qs = self.q_net(torch.tensor(state).to(self.device).view(-1,4))
        action = qs.argmax().detach().cpu().numpy()
        
        ############################
        return action

    @torch.no_grad()
    def get_Q_target(self, state, action, reward, done, next_state) -> torch.Tensor:
        """
        Get the target Q value according to the Bellman equation
        """
        if self.use_double:
            # YOUR IMPLEMENTATION HERE
            a = self.q_net(next_state.to(self.device)).argmax(dim=1, keepdim=True).detach()
            # print(a.size(), self.target_net(next_state.to(self.device)).size())
            a = torch.tensor(a, dtype=torch.int64).reshape(-1, 1).to(self.device)
            next_q = self.target_net(next_state.to(self.device)).gather(1, a).view(-1)
            q_target = reward.to(self.device) + (1 - done.to(self.device)) * self.gamma * next_q
        else:
            # YOUR IMPLEMENTATION HERE
            next_q_max = self.target_net(next_state.to(self.device)).max(dim=1)[0]
            q_target = reward.to(self.device) + (1 - done.to(self.device)) * self.gamma * next_q_max

        return q_target

    def get_Q(self, state, action, use_double_net=False) -> torch.Tensor:
        """
        Get the Q value of the current state and action
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        #action = torch.tensor(action, dtype=torch.int64).reshape(-1, 1).to(self.device)
        # print(action.size())
        q = self.q_net(state.to(self.device)).gather(1, action.to(self.device).long()).view(-1)
        return q
        ############################

    def update(self, batch, step, weights=None):
        state, action, reward, next_state, done = batch

        Q_target = self.get_Q_target(state, action, reward, done, next_state)

        Q = self.get_Q(state, action)

        if weights is None:
            weights = torch.ones_like(Q).to(self.device)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not step % self.target_update_interval:
            with torch.no_grad():
                self.soft_update(self.target_net, self.q_net)

        return loss.item(), td_error, Q.mean().item()

    def save(self, name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join('models', name))

    def load(self, name='best.pt'):
        self.q_net.load_state_dict(torch.load(os.path.join('models', name)))

    def __repr__(self) -> str:
        use_double = 'Double' if self.use_double else ''
        use_dueling = 'Dueling' if self.use_dueling else ''
        prefix = 'Normal' if not self.use_double and not self.use_dueling else ''
        return use_double + use_dueling + prefix + 'QNetwork'
