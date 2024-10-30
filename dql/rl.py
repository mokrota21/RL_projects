from base import State, DEVICE, torch
from environment import Agent, Environment, Point, Policy, ACTIONS, deque, np
import torch.nn as nn
from random import random, sample, choice
import matplotlib as plt
from datetime import datetime
from typing import List

# Pros: Relatively simple. Cons: May be not representative enough
def one_hidden(input_size, dropout_rate):
    network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, len(ACTIONS))
        )
    return network

# Pros: Can represent for sure. Cons: May be too complicated.
def exp_hidden(input_size, dropout_rate):
    network = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(100, 81),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(81, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 49),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(49, 36),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(36, 25),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(25, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(16, 9),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(9, len(ACTIONS))
        )
    return network

def small_one_hidden(input_size, dropout_rate):
    network = nn.Sequential(
            nn.Linear(input_size, 6),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4, len(ACTIONS))
        )
    return network

# Linear approximator
def linear(input_size, dropout_rate):
    network = nn.Sequential(
            nn.Linear(input_size, len(ACTIONS))
        )
    return network

ARCHITECTURES = {
    'one_hidden': one_hidden,
    'exp_hidden': exp_hidden,
    'small_one_hidden': small_one_hidden,
    'linear': linear
}

class DQNetwork(nn.Module):
    """Neural Network structure for value action function approximation Q"""

    def __init__(self, input_size, architecture: str, dropout_rate = 0.2):
        super().__init__()
        self.network = ARCHITECTURES[architecture](input_size, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQL, optimized to store tensors for faster sampling."""
    
    def __init__(self, max_size: int, device: torch.device = DEVICE):
        self.buffer = []
        self.max_size = max_size
        self.device = device
        
    def push(self, state: State, action: int, reward: float, next_state: State, terminate: bool):
        # Convert each element to a tensor and move it to the specified device
        state_tensor = state.to_tensor().to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state_tensor = next_state.to_tensor().to(self.device)
        terminate_tensor = torch.tensor(not terminate, dtype=torch.int64, device=self.device)
        
        # Store as a tuple of tensors
        self.buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, terminate_tensor))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        
    def get_batch(self, batch_size: int):
        batch = sample(self.buffer, batch_size)
        # Unzip and stack tensors for batch processing
        states, actions, rewards, next_states, terminates = zip(*batch)
        
        # Stack directly without further conversion
        state_tensor = torch.stack(states)
        action_tensor = torch.stack(actions).unsqueeze(1)
        reward_tensor = torch.stack(rewards).unsqueeze(1)
        next_state_tensor = torch.stack(next_states)
        terminate_tensor = torch.stack(terminates).unsqueeze(1)
        
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, terminate_tensor

# class ReplayBuffer:
#     """Experience replay buffer for DQL to avoid being stuck in local minima"""
    
#     def __init__(self, max_size: int):
#         self.buffer = list()
#         self.max_size = max_size
        
#     def push(self, state: State, action: int, reward: float, next_state: State, terminate: bool):
#         self.buffer.append((state, action, reward, next_state, terminate))
#         if len(self.buffer) > self.max_size:
#             self.buffer.pop(0)
        
#     def get_batch(self, batch_size: int) -> List[tuple]:
#         return sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class ValueAction:
    """Implementation of value action function Q with Neural Network."""

    def __init__(self, state_size, architecture, batch_size=64, buffer_size=10000, steps_per_update=100, dim=0.99, alpha=0.001, tau=0.001, dropout=0.2):
        "To train existing model we give model as input. It is assumed that output of model is of size of 4"
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.dim = dim
        self.tau = tau
        self.steps_per_update = steps_per_update
        self.step = 0

        # DQ networks. We had some bugs with using only one network so for now decided to split them
        self.qnetwork_online = DQNetwork(state_size, architecture=architecture, dropout_rate=dropout).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.qnetwork_online.parameters(), lr=alpha)
        # self.optimizer = torch.optim.SGD(self.qnetwork_online.parameters(), lr=alpha)
        self.qnetwork_other = DQNetwork(state_size, architecture=architecture, dropout_rate=dropout).to(DEVICE)
        self.qnetwork_other.load_state_dict(self.qnetwork_online.state_dict())
    
    def learn(self):
        """Update Q-Network weights using batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sampling actions
        batch = self.memory.get_batch(self.batch_size)
        state_tensor, action_tensor, reward_tensor, next_state_tensor, terminate_tensor = batch
        # states, actions, rewards, next_states, terminate = zip(*batch)
        
        # # Converting to tensors
        # s = datetime.now()
        # state_tensor = torch.stack([s.to_tensor() for s in states]).to(DEVICE)
        # print(datetime.now() - s)
        # action_tensor = torch.tensor(actions).unsqueeze(1).to(DEVICE)
        # reward_tensor = torch.tensor(rewards).unsqueeze(1).to(DEVICE)
        # next_state_tensor = torch.stack([s.to_tensor() for s in next_states]).to(DEVICE)

        # Computing q values
        current_q_values = self.qnetwork_online(state_tensor)
        current_q_values = current_q_values.gather(1, action_tensor)
        with torch.no_grad():
            next_q_values = self.qnetwork_other(next_state_tensor)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        target_q_values = reward_tensor + self.dim * next_q_values
        
        # Optimize
        loss = nn.MSELoss()(current_q_values.float(), target_q_values.detach().float())
        # loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        if self.step % self.steps_per_update:
            self.update_other()
            self.step = 0
        
        return loss.item()
    
    def update_other(self):
        """Copy paste weights"""
        for other_param, online_param in zip(self.qnetwork_other.parameters(), self.qnetwork_online.parameters()):
            other_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * other_param.data)

    def save(self, path: str):
        """Save weights"""
        torch.save({
            'online_state_dict': self.qnetwork_online.state_dict(),
            'other_state_dict': self.qnetwork_other.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load weights"""
        weights = torch.load(path)
        self.qnetwork_online.load_state_dict(weights['online_state_dict'])
        self.qnetwork_other.load_state_dict(weights['other_state_dict'])

class EPolicy(Policy):
    "Epsilon policy, communicates agent with Value Action function"
    def __init__(self, value_action: ValueAction, initial_epsilon = 0.1, decay = 0.999):
        self.value_action = value_action
        self.epsilon = initial_epsilon
        self.decay = decay
    
    def next_action(self, agent: Agent):
        "Take next action greedy and with probability epsilon take random valid action (bumping in the wall randomly wouldn't benefit)"

        state = agent.get_state()
        env = agent.environment
        pos = agent.pos_history[-1]

        if random() < self.epsilon:
            return choice([0, 1, 2, 3])
            # return choice(env.valid_actions(pos=pos))
        with torch.no_grad():
            state_tensor = state.to_tensor()
            action_values = self.value_action.qnetwork_online(state_tensor)
            return torch.argmax(action_values).item()
    
    def decay_epsilon(self):
        self.epsilon = self.epsilon * self.decay
