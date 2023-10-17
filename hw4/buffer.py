import torch
import numpy as np
from collections import deque


def get_buffer(cfg, **args):
    assert type(cfg.nstep) == int and cfg.nstep > 0, 'nstep must be a positive integer'
    if cfg.use_ppo:
        return PPOReplayBuffer(cfg.capacity, **args, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, num_envs=cfg.vec_envs)
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
    def __init__(self, capacity, state_size, action_size, device, seed):
        self.device = device
        self.state = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.action = torch.zeros(capacity, action_size, dtype=torch.float).contiguous().pin_memory()
        self.reward = torch.zeros(capacity, dtype=torch.float).contiguous().pin_memory()
        self.next_state = torch.zeros(capacity, state_size, dtype=torch.float).contiguous().pin_memory()
        self.done = torch.zeros(capacity, dtype=torch.int).contiguous().pin_memory()
        self.rng = np.random.default_rng(seed)
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
        # raise NotImplementedError
        ############################

    def sample(self, batch_size):
        # using np.random.default_rng().choice is faster https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        batch = ()
        # get a batch of data from the buffer according to the sample_idxs
        # please transfer the data to the corresponding device before return
        ############################
        # YOUR IMPLEMENTATION HERE #
        batch = (self.state[sample_idxs], self.action[sample_idxs], self.reward[sample_idxs], 
                 self.next_state[sample_idxs], self.done[sample_idxs])

        # raise NotImplementedError
        ############################
        return batch


class PPOReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_size, action_size, device, seed, gamma, gae_lambda, num_envs):
        self.device = device
        capacity = capacity // num_envs
        self.state = torch.zeros(capacity, num_envs, state_size, dtype=torch.float).contiguous().pin_memory()
        self.action = torch.zeros(capacity, num_envs, action_size, dtype=torch.float).contiguous().pin_memory()
        self.reward = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
        self.next_state = torch.zeros(capacity, num_envs, state_size, dtype=torch.float).contiguous().pin_memory()
        self.done = torch.zeros(capacity, num_envs, dtype=torch.int).contiguous().pin_memory()
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity

        self.advantage = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
        self.value = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
        self.log_prob = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()
        self.returns = torch.zeros(capacity, num_envs, dtype=torch.float).contiguous().pin_memory()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_envs = num_envs

        self.to_device()
    
    def to_device(self):
        self.state = self.state.to(self.device)
        self.action = self.action.to(self.device)
        self.reward = self.reward.to(self.device)
        self.next_state = self.next_state.to(self.device)
        self.done = self.done.to(self.device)
        self.advantage = self.advantage.to(self.device)
        self.value = self.value.to(self.device)
        self.log_prob = self.log_prob.to(self.device)
        self.returns = self.returns.to(self.device)
    
    def __repr__(self) -> str:
        return 'PPOReplayBuffer'
    
    def add(self, transition):
        state, action, reward, next_state, done, value, log_prob = transition
        super().add((state, action, reward, next_state, done))

        tmp_idx = self.idx - 1 if self.idx > 0 else self.capacity - 1
        self.value[tmp_idx] = torch.as_tensor(value)
        self.log_prob[tmp_idx] = torch.as_tensor(log_prob)
    
    def clear(self):
        self.idx = 0
        self.size = 0
    
    def make_dataset(self):
        batch = (
            self.state[:self.size].flatten(0, 1),
            self.action[:self.size].flatten(0, 1),
            self.log_prob[:self.size].flatten(0, 1),
            self.value[:self.size].flatten(0, 1),
            self.advantage[:self.size].flatten(0, 1),
            self.returns[:self.size].flatten(0, 1)
        )
        return batch
    
    def get_next_values(self, agent, t) -> torch.Tensor:
        """
        Given timestep t and the current agent, obtain or calculate values of t + 1
        """
        # If t is the last timestep or is done, return the value of the next state from the agent
        # Otherwise, you can directly return the value of the next state from the buffer

        # You can assume that the buffer is full, and vector envs are used.
        ############################
        # YOUR IMPLEMENTATION HERE #
        if t < self.capacity - 1:
            next_values = self.value[t + 1]
        else:
            next_values = agent.get_value(self.next_state[t])
        # raise NotImplementedError
        ############################
        return torch.tensor(next_values).to(self.device)
    

    def compute_advantages_and_returns(self, agent) -> None:
        """
        Once the buffer is full, calculate all the advantages and returns for each timestep
        """
        for t in reversed(range(self.capacity)):
            next_values = self.get_next_values(agent, t)
            # use the formula for GAE-lambda to calculate delta
            # use delta to calculate the advantage
            # you can directly update the advantage in the buffer
            # Hint: can you calculate step t's advantage using step t + 1's advantage?
            ############################
            # YOUR IMPLEMENTATION HERE #
            # self.returns[t] = self.reward[t] + self.gamma * self.returns[t+1]
            # print(self.advantage[t].size(), next_values.size(), self.reward.size(), self.value.size())
            self.advantage[t] = self.reward[t] + self.gamma * next_values * (1 - self.done[t]) - self.value[t]
            if t < self.capacity - 1: 
                self.advantage[t] += self.gamma * self.gae_lambda * self.advantage[t+1]
            ############################
        # calculate the returns
        ############################
        # YOUR IMPLEMENTATION HERE #
        self.returns = self.advantage + self.value
        # raise NotImplementedError
        ############################


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, action_size, device, seed):
        super().__init__(capacity, state_size, action_size, device, seed)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        """Get n-step state, action, reward and done forwards, break if there's a done"""
        ############################
        # （OPTIONAL) YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
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
    def __init__(self, capacity, eps, alpha, beta, state_size, action_size, device, seed):
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, action_size, device, seed)

    def add(self, transition):
        """
        Add a new experience to memory, and update it's priority to the max_priority.
        """
        ############################
        # （OPTIONAL)  YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer with priority, and calculates the weights used for the correction of bias used in the Q-learning update
        Returns:
            batch: a batch of experiences as in the normal replay buffer
            weights: torch.Tensor (batch_size, ), importance sampling weights for each sample
            sample_idxs: numpy.ndarray (batch_size, ), the indexes of the sample in the buffer
        """
        ############################
        # （OPTIONAL)  YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha

        self.priorities[data_idxs] = priorities
        self.max_priority = np.max(self.priorities)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'


# Avoid Diamond Inheritance
class PrioritizedNStepReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, action_size, device, seed):
        ############################
        # （OPTIONAL)  YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################

    def __repr__(self) -> str:
        return f'Prioritized{self.n_step}StepReplayBuffer'

    def add(self, transition):
        ############################
        # （OPTIONAL)  YOUR IMPLEMENTATION HERE #

        raise NotImplementedError
        ############################

    # def the other necessary class methods as your need