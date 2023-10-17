import torch
import numpy as np
from collections import deque


def get_buffer(cfg, **args):
    assert type(cfg.nstep) == int and cfg.nstep > 0, 'nstep must be a positive integer'
    if not cfg.use_per:
        if cfg.nstep == 1:
            return ReplayBuffer(cfg.capacity, **args)
        else:
            return NStepReplayBuffer(cfg.capacity, cfg.nstep, cfg.gamma, **args)
    else:
        if cfg.nstep == 1:
            return PrioritizedReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, **args)
        else:
            return PrioritizedNStepReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, cfg.nstep, cfg.gamma, **args)


class ReplayBuffer:
    def __init__(self, capacity, state_size, device):
        self.device = device
        self.state = torch.empty(capacity, state_size, dtype=torch.float)
        self.action = torch.empty(capacity, 1, dtype=torch.float)
        self.reward = torch.empty(capacity, dtype=torch.float)
        self.next_state = torch.empty(capacity, state_size, dtype=torch.float)
        self.done = torch.empty(capacity, dtype=torch.int)

        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer and update the index and size of the buffer
        # you may need to convert the data type to torch.tensor
        
        ############################
        # YOUR IMPLEMENTATION HERE #
        idx = self.idx
        self.state[idx].copy_(torch.tensor(state))
        self.action[idx].copy_(torch.tensor(action))
        self.reward[idx].copy_(torch.tensor(reward))
        self.next_state[idx].copy_(torch.tensor(next_state))
        self.done[idx].copy_(torch.tensor(done))
        self.size = min(self.size + 1, self.capacity)
        self.idx = (idx + 1) % self.capacity

            
        ############################

    def sample(self, batch_size):
        # sample batch_size data from the buffer without replacement
        sample_idxs = np.random.choice(self.size, batch_size, replace=False)
        batch = ()
        # get a batch of data from the buffer according to the sample_idxs
        # please transfer the data to the corresponding device before return
        ############################
        # YOUR IMPLEMENTATION HERE #
        batch = (self.state[sample_idxs], self.action[sample_idxs], self.reward[sample_idxs], 
                 self.next_state[sample_idxs], self.done[sample_idxs])

        ############################
        return batch


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, device):
        super().__init__(capacity, state_size, device=device)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        """Get n-step state, action, reward and done for the transition, discard those rewards after done=True"""
        ############################
        # YOUR IMPLEMENTATION HERE #
        state, action, reward, done = self.n_step_buffer[0]
        gamma = 1
        for i in range(1, self.n_step):
            gamma *= self.gamma
            _, __, reward_i, ___ = self.n_step_buffer[i]
            reward += gamma * reward_i
        ############################
        return state, action, reward, done

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, device):
        self.weights = np.zeros(capacity, dtype=np.float32) # stores weights for importance sampling
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, device=device)

    def add(self, transition):
        """
        Add a new experience to memory, and update it's priority to the max_priority.
        """
        state, action, reward, next_state, done = transition 
        idx = self.idx
        self.weights[idx] = (self.max_priority + self.eps) ** self.alpha
        ReplayBuffer.add(self, (state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer with priority, and calculates the weights used for the correction of bias used in the Q-learning update
        Returns:
            batch: a batch of experiences as in the normal replay buffer
            weights: torch.Tensor (batch_size, ), importance sampling weights for each sample
            sample_idxs: numpy.ndarray (batch_size, ), the indexes of the sample in the buffer
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        p = self.weights[:self.idx]
        p = p / p.sum()
        sample_idxs = np.random.choice(self.size, batch_size, replace=False, p=p)
        batch = (self.state[sample_idxs], self.action[sample_idxs], self.reward[sample_idxs], 
                 self.next_state[sample_idxs], self.done[sample_idxs])
        weights = p[sample_idxs]
        w = (self.size * weights) ** (-self.beta)
        w = w / w.max()       
        weights = torch.tensor(weights * w).to(self.device)
        ############################
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha

        self.weights[data_idxs] = priorities
        self.max_priority = max(self.weights)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'


# Avoid Diamond Inheritance
class PrioritizedNStepReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, device):
        ############################
        # YOUR IMPLEMENTATION HERE #
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque([], maxlen=n_step)
        super().__init__(capacity, state_size, device=device, alpha=alpha, beta=beta, state_size=state_size)

        ############################
    def __repr__(self) -> str:
        return f'Prioritized{self.n_step}StepReplayBuffer'

    def add(self, transition):
        ############################
        # YOUR IMPLEMENTATION HERE #
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_buffer[0]
        gamma = 1
        for i in range(1, self.n_step):
            gamma *= self.gamma
            _, __, reward_i, ___ = self.n_step_buffer[i]
            reward += gamma * reward_i
        PrioritizedReplayBuffer.add(self, (state, action, reward, next_state, done))
        
        ############################

    # def the other necessary class methods as your need