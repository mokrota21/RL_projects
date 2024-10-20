from base import State, DEVICE, torch
from environment import Agent, Environment, Point, Policy, ACTIONS, deque, np
import torch.nn as nn
from random import random, sample, choice
import matplotlib as plt
from copy import deepcopy
from typing import List

class DQNetwork(nn.Module):
    """Neural Network structure for value action function approximation Q"""

    def __init__(self, input_size: int, dropout_rate = 0.2):
        super().__init__()
        # self.network = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, len(ACTIONS))
        # )
        # self.network = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(16, 8),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(8, len(ACTIONS))
        # )
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 25),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(25, len(ACTIONS))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQL to avoid being stuck in local minima"""
    
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
        
    def push(self, state: State, action: int, reward: float, next_state: State):
        self.buffer.append((state, action, reward, next_state))
        
    def get_batch(self, batch_size: int) -> List[tuple]:
        return sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class ValueAction:
    """Implementation of value action function Q with Neural Network."""

    def __init__(self, state_size, batch_size=64, buffer_size=10000, dim=0.99, alpha=0.001, tau=0.001, dropout=0.2):
        "To train existing model we give model as input. It is assumed that output of model is of size of 4"
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.dim = dim
        self.tau = tau

        # DQ networks. We had some bugs with using only one network so for now decided to split them
        self.qnetwork_online = DQNetwork(state_size, dropout_rate=dropout).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.qnetwork_online.parameters(), lr=alpha)
        # self.optimizer = torch.optim.SGD(self.qnetwork_online.parameters(), lr=alpha)
        self.qnetwork_other = DQNetwork(state_size, dropout_rate=dropout).to(DEVICE)
        self.qnetwork_other.load_state_dict(self.qnetwork_online.state_dict())
    
    def learn(self):
        """Update Q-Network weights using batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sampling actions
        batch = self.memory.get_batch(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Converting to tensors
        state_tensor = torch.stack([s.to_tensor() for s in states]).to(DEVICE)
        action_tensor = torch.tensor(actions).unsqueeze(1).to(DEVICE)
        reward_tensor = torch.tensor(rewards).unsqueeze(1).to(DEVICE)
        next_state_tensor = torch.stack([s.to_tensor() for s in next_states]).to(DEVICE)
        
        # Computing q values
        current_q_values = self.qnetwork_online(state_tensor)
        current_q_values = current_q_values.gather(1, action_tensor)
        with torch.no_grad():
            next_q_values = self.qnetwork_other(next_state_tensor)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        target_q_values = reward_tensor + self.dim * next_q_values
        
        # Optimize
        # loss = nn.MSELoss()(current_q_values, target_q_values)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_other()
        
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
