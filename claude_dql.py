import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

torch.set_default_device('cuda')

# Constants
EMPTY = 0
SUBGOAL = -1
GOAL = -2
WALL = -3
UNOBSERVED = -4
AGENT = -5
VISIBILITY = 2  # Radius of visibility around agent
DEVICE = 'cuda'


maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, SUBGOAL, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, GOAL, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
replace_1 = lambda x: WALL if x == 1 else x
maze_map = list(map(lambda x: list(map(replace_1, x)), maze_map))
maze_map = np.array(maze_map, dtype=np.int64)

@dataclass
class Position:
    y: int
    x: int
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.y + other.y, self.x + other.x)
    
    def as_tuple(self) -> Tuple[int, int]:
        return (self.y, self.x)

class Environment:
    """Handles the maze environment and its dynamics"""
    
    def __init__(self, maze_map: np.ndarray):
        self.maze_map = maze_map
        self.shape = maze_map.shape
        
    def is_valid_position(self, pos: Position) -> bool:
        return (0 <= pos.y < self.shape[0] and 
                0 <= pos.x < self.shape[1] and 
                self.maze_map[pos.y, pos.x] != WALL)
    
    def get_reward(self, pos: Position, has_subgoal: bool) -> float:
        tile = self.maze_map[pos.y, pos.x]
        if tile == WALL:
            return -10.0
        elif tile == SUBGOAL and not has_subgoal:
            return 5.0
        elif tile == GOAL and has_subgoal:
            return 10.0
        return -0.1  # Small negative reward to encourage finding goal quickly

    def get_valid_actions(self, pos: Position) -> List[int]:
        actions = []
        for idx, action in enumerate(ACTIONS):
            new_pos = pos + action
            if self.is_valid_position(new_pos):
                actions.append(idx)
        return actions

    def get_random_empty_position(self) -> Position:
        empty_positions = []
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                if self.maze_map[y, x] == EMPTY:
                    empty_positions.append(Position(y, x))
        return random.choice(empty_positions)

# Define possible actions as Position objects
ACTIONS = [
    Position(-1, 0),  # Up
    Position(1, 0),   # Down
    Position(0, 1),   # Right
    Position(0, -1)   # Left
]

class State:
    """Represents the current state of the agent"""
    
    def __init__(self, observation: np.ndarray, position: Position, has_subgoal: bool):
        self.observation = observation
        self.position = position
        self.has_subgoal = has_subgoal
        
    def to_tensor(self) -> torch.Tensor:
        # Flatten observation and concatenate with position and subgoal flag
        obs_flat = self.observation.flatten()
        pos_y, pos_x = self.position.as_tuple()
        state_array = np.concatenate([
            obs_flat,
            np.array([pos_y, pos_x, float(self.has_subgoal)])
        ])
        # print(len(state_array))
        return torch.FloatTensor(state_array).to(DEVICE)

class QNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(ACTIONS))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQL"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: State, action: int, reward: float, 
             next_state: State, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQLAgent:
    """Deep Q-Learning agent implementation"""
    
    def __init__(self, 
                 env: Environment,
                 state_size: int,
                 hidden_size: int = 128,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 learning_rate: float = 0.0015,
                 epsilon_start: float = 0.6,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.9999):
        
        self.env = env
        self.device = torch.device(DEVICE)
        
        # Q-Networks (online and target)
        self.qnetwork_online = QNetwork(state_size, hidden_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, hidden_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_online.state_dict())
        
        self.optimizer = torch.optim.Adam(self.qnetwork_online.parameters(), 
                                        lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        # Training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.001  # Soft update parameter
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State tracking
        self.position = None
        self.has_subgoal = False
        self.observation = None
        self.steps = 0
        
    def reset(self, start_pos: Optional[Position] = None):
        """Reset the agent to initial state"""
        self.position = start_pos or self.env.get_random_empty_position()
        self.has_subgoal = False
        self.observation = np.full(self.env.shape, UNOBSERVED)
        self.update_observation()
        return self.get_state()
    
    def update_observation(self):
        """Update the agent's observation based on visibility radius"""
        pos_y, pos_x = self.position.as_tuple()
        
        # Calculate visibility bounds
        y_min = max(0, pos_y - VISIBILITY)
        y_max = min(self.env.shape[0], pos_y + VISIBILITY + 1)
        x_min = max(0, pos_x - VISIBILITY)
        x_max = min(self.env.shape[1], pos_x + VISIBILITY + 1)
        
        # Update visible area
        self.observation[y_min:y_max, x_min:x_max] = self.env.maze_map[y_min:y_max, x_min:x_max]
    
    def get_state(self) -> State:
        """Get current state representation"""
        return State(self.observation, self.position, self.has_subgoal)
    
    def select_action(self, state: State) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0)
                action_values = self.qnetwork_online(state_tensor)
                return torch.argmax(action_values).item()
        return random.choice(self.env.get_valid_actions(state.position))
    
    def step(self, action: int) -> Tuple[State, float, bool]:
        """Take action and return new state, reward, and done flag"""
        new_position = self.position + ACTIONS[action]
        reward = self.env.get_reward(new_position, self.has_subgoal)
        
        if self.env.is_valid_position(new_position):
            self.position = new_position
            tile = self.env.maze_map[new_position.y, new_position.x]
            
            if tile == SUBGOAL:
                self.has_subgoal = True
            elif tile == GOAL and self.has_subgoal:
                return self.get_state(), reward, True
        
        self.update_observation()
        return self.get_state(), reward, False
    
    def learn(self):
        """Update Q-Network weights using batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_tensor = torch.stack([s.to_tensor() for s in states])
        action_tensor = torch.tensor(actions).unsqueeze(1)
        reward_tensor = torch.tensor(rewards).unsqueeze(1)
        next_state_tensor = torch.stack([s.to_tensor() for s in next_states])
        done_tensor = torch.tensor(dones, dtype=torch.int32).unsqueeze(1)
        
        # Compute Q values
        prediction_tensor = self.qnetwork_online(state_tensor)
        current_q_values = prediction_tensor.gather(1, action_tensor)
        next_q_values = self.qnetwork_target(next_state_tensor).max(1)[0].unsqueeze(1)
        target_q_values = reward_tensor + self.gamma * next_q_values * (1 - done_tensor)
        
        # Compute loss and update online network
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _soft_update(self):
        """Soft update of target network parameters"""
        for target_param, online_param in zip(
            self.qnetwork_target.parameters(),
            self.qnetwork_online.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'online_state_dict': self.qnetwork_online.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.qnetwork_online.load_state_dict(checkpoint['online_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_agent(agent: DQLAgent, 
                num_episodes: int,
                max_steps: int = 100,
                print_every: int = 1) -> List[float]:
    """Train the agent and return episode rewards"""
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = agent.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = agent.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.learn()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return episode_rewards


import pyxel
import numpy as np
from typing import Optional

class MazeVisualizer:
    def __init__(self, env, agent, cell_size=16):
        self.env = env
        self.agent = agent
        self.cell_size = cell_size
        
        # Calculate window dimensions
        self.width = self.env.shape[1] * cell_size
        self.height = self.env.shape[0] * cell_size
        
        # Initialize Pyxel
        pyxel.init(self.width, self.height, fps=10, title="Maze Agent Visualization")
        
        # Color mappings
        self.colors = {
            EMPTY: 7,      # White
            WALL: 0,       # Black
            SUBGOAL: 9,    # Orange
            GOAL: 11,      # Yellow
            UNOBSERVED: 5, # Dark blue
            AGENT: 8       # Red
        }
        
        # Initialize agent state
        self.current_state = None
        self.done = False
        self.steps = 0
        self.max_steps = 1000
        
        # Start Pyxel
        pyxel.run(self.update, self.draw)
    
    def reset_simulation(self):
        self.current_state = self.agent.reset()
        self.done = False
        self.steps = 0
    
    def update(self):
        if pyxel.btnp(pyxel.KEY_R):
            self.reset_simulation()
            
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()
            
        if not self.done and self.current_state is not None and self.steps < self.max_steps:
            action = self.agent.select_action(self.current_state)
            next_state, reward, self.done = self.agent.step(action)
            self.current_state = next_state
            self.steps += 1
            
        if self.current_state is None:
            self.reset_simulation()
    
    def draw(self):
        pyxel.cls(0)
        
        # Draw maze
        for y in range(self.env.shape[0]):
            for x in range(self.env.shape[1]):
                cell_value = self.current_state.observation[y, x]
                color = self.colors.get(cell_value, 0)
                
                pyxel.rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                    color
                )
        
        # Draw agent
        agent_y, agent_x = self.current_state.position.as_tuple()
        pyxel.rect(
            agent_x * self.cell_size,
            agent_y * self.cell_size,
            self.cell_size,
            self.cell_size,
            self.colors[AGENT]
        )
        
        # Draw status
        status_color = 11 if self.current_state.has_subgoal else 8
        pyxel.rect(0, 0, 4, 4, status_color)

# Function to run the visualization
def visualize_agent(env, agent):
    visualizer = MazeVisualizer(env, agent)

import pyxel
import numpy as np
from typing import Optional, List, Deque
from collections import deque

class TrainingVisualizer:
    def __init__(self, env, agent, window_size=200, cell_size=16):
        self.env = env
        self.agent = agent
        self.cell_size = cell_size
        self.window_size = window_size
        
        # Calculate window dimensions
        self.maze_width = self.env.shape[1] * cell_size
        self.maze_height = self.env.shape[0] * cell_size
        self.metrics_height = 120  # Height for metrics visualization
        
        # Initialize Pyxel with larger window
        pyxel.init(
            max(self.maze_width, self.window_size), 
            self.maze_height + self.metrics_height, 
            title="Maze Agent Training Visualization"
        )
        
        # Color mappings
        self.colors = {
            EMPTY: 7,      # White
            WALL: 0,       # Black
            SUBGOAL: 9,    # Orange
            GOAL: 11,      # Yellow
            UNOBSERVED: 5, # Dark blue
            AGENT: 8       # Red
        }
        
        # Training metrics
        self.episode_rewards = deque(maxlen=window_size)
        self.avg_rewards = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.success_rate = deque(maxlen=window_size)
        
        # Episode tracking
        self.current_episode = 0
        self.episodes_completed = 0
        self.current_state = None
        self.done = False
        self.steps = 0
        self.max_steps = 100
        self.episode_reward = 0
        self.current_loss = 0
        self.training_complete = False
        
        # Training parameters
        self.num_episodes = 1000
        self.print_every = 10
        
        # Start Pyxel
        pyxel.run(self.update, self.draw)
    
    def reset_episode(self):
        self.current_state = self.agent.reset()
        self.done = False
        self.steps = 0
        self.episode_reward = 0
    
    def update(self):
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()
        
        if not self.training_complete:
            # Training loop
            if self.current_state is None or self.done or self.steps >= self.max_steps:
                if self.current_state is not None:
                    # Record metrics
                    self.episode_rewards.append(self.episode_reward)
                    if len(self.episode_rewards) >= 10:
                        self.avg_rewards.append(np.mean(list(self.episode_rewards)[-10:]))
                    self.success_rate.append(1.0 if self.done else 0.0)
                
                self.episodes_completed += 1
                if self.episodes_completed >= self.num_episodes:
                    self.training_complete = True
                    return
                
                self.reset_episode()
                return
            
            # Take a step in the environment
            action = self.agent.select_action(self.current_state)
            next_state, reward, self.done = self.agent.step(action)
            
            # Store transition and learn
            self.agent.memory.push(self.current_state, action, reward, next_state, self.done)
            loss = self.agent.learn()
            
            if loss is not None:
                self.losses.append(loss)
                self.current_loss = loss
            
            self.current_state = next_state
            self.episode_reward += reward
            self.steps += 1
    
    def draw(self):
        pyxel.cls(0)
        
        # Draw maze
        if self.current_state is not None:
            for y in range(self.env.shape[0]):
                for x in range(self.env.shape[1]):
                    cell_value = self.current_state.observation[y, x]
                    color = self.colors.get(cell_value, 0)
                    
                    pyxel.rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                        color
                    )
            
            # Draw agent
            agent_y, agent_x = self.current_state.position.as_tuple()
            pyxel.rect(
                agent_x * self.cell_size,
                agent_y * self.cell_size,
                self.cell_size,
                self.cell_size,
                self.colors[AGENT]
            )
            
            # Draw subgoal status
            status_color = 11 if self.current_state.has_subgoal else 8
            pyxel.rect(0, 0, 4, 4, status_color)
        
        # Draw metrics
        metrics_y = self.maze_height + 10
        
        # Draw episode progress
        progress = f"Episode: {self.episodes_completed}/{self.num_episodes}"
        pyxel.text(10, metrics_y, progress, 7)
        
        # Draw current episode reward
        reward_text = f"Current Reward: {self.episode_reward:.1f}"
        pyxel.text(10, metrics_y + 10, reward_text, 7)
        
        # Draw current loss
        loss_text = f"Current Loss: {self.current_loss:.3f}"
        pyxel.text(10, metrics_y + 20, loss_text, 7)
        
        # Draw average reward graph
        if len(self.avg_rewards) > 1:
            self._draw_graph(
                self.avg_rewards,
                180,  # x position
                metrics_y,
                "Avg Reward",
                color=11,
                min_val=min(self.avg_rewards),
                max_val=max(self.avg_rewards)
            )
        
        # Draw success rate graph
        if len(self.success_rate) > 1:
            self._draw_graph(
                self.success_rate,
                180,
                metrics_y + 60,
                "Success Rate",
                color=3,
                min_val=0,
                max_val=1
            )
    
    def _draw_graph(self, data, x, y, title, color, min_val, max_val):
        # Draw title
        pyxel.text(x, y, title, 7)
        
        # Draw graph
        graph_width = min(len(data), self.window_size - 200)
        graph_height = 40
        
        # Draw axes
        pyxel.line(x, y + 10, x + graph_width, y + 10, 1)  # x-axis
        pyxel.line(x, y + 10, x, y + 10 - graph_height, 1)  # y-axis
        
        # Draw data points
        for i in range(len(data) - 1):
            if i >= graph_width - 1:
                break
                
            val1 = data[i]
            val2 = data[i + 1]
            
            # Normalize values to graph height
            y1 = y + 10 - int((val1 - min_val) / (max_val - min_val) * graph_height)
            y2 = y + 10 - int((val2 - min_val) / (max_val - min_val) * graph_height)
            
            pyxel.line(x + i, y1, x + i + 1, y2, color)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pygame
import numpy as np
from collections import deque
import threading
from queue import Queue


class MetricsVisualizer:
    def __init__(self, window_size=100000):
        self.window_size = window_size
        
        # Metrics storage
        self.losses = deque(maxlen=window_size)
        self.avg_rewards = deque(maxlen=window_size)
        self.success_rates = deque(maxlen=window_size)
        self.epsilon_values = deque(maxlen=window_size)
        
        # Setup pygame
        pygame.init()
        self.plot_size = (800, 600)
        self.screen = pygame.display.set_mode(self.plot_size)
        pygame.display.set_caption('Training Metrics')
        
        # Setup the plot
        plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 6), dpi=100)
        self.fig.suptitle('Training Metrics', fontsize=16)
        
        # Initialize lines
        self.lines = {}
        self.setup_subplots()
        
        # Communication queue for thread-safe updates
        self.metrics_queue = Queue()
        self.running = True
        
        # For converting matplotlib to pygame surface
        self.canvas = agg.FigureCanvasAgg(self.fig)
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def setup_subplots(self):
        # Loss subplot
        self.axs[0, 0].set_title('Training Loss')
        self.axs[0, 0].set_xlabel('Steps')
        self.axs[0, 0].set_ylabel('Loss')
        self.lines['loss'], = self.axs[0, 0].plot([], [], 'r-', label='Loss')
        self.axs[0, 0].grid(True, alpha=0.3)
        
        # Average reward subplot
        self.axs[0, 1].set_title('Average Reward')
        self.axs[0, 1].set_xlabel('Episodes')
        self.axs[0, 1].set_ylabel('Reward')
        self.lines['reward'], = self.axs[0, 1].plot([], [], 'g-', label='Avg Reward')
        self.axs[0, 1].grid(True, alpha=0.3)
        
        # Success rate subplot
        self.axs[1, 0].set_title('Success Rate')
        self.axs[1, 0].set_xlabel('Episodes')
        self.axs[1, 0].set_ylabel('Rate')
        self.lines['success'], = self.axs[1, 0].plot([], [], 'b-', label='Success Rate')
        self.axs[1, 0].grid(True, alpha=0.3)
        
        # Epsilon subplot
        self.axs[1, 1].set_title('Exploration Rate (ε)')
        self.axs[1, 1].set_xlabel('Episodes')
        self.axs[1, 1].set_ylabel('ε')
        self.lines['epsilon'], = self.axs[1, 1].plot([], [], 'y-', label='ε')
        self.axs[1, 1].grid(True, alpha=0.3)
        
        # Add legends
        for ax in self.axs.flat:
            ax.legend()
        
        plt.tight_layout()
    
    def update_metrics(self, loss=None, reward=None, success=None, epsilon=None):
        """Thread-safe method to update metrics"""
        self.metrics_queue.put({
            'loss': loss,
            'reward': reward,
            'success': success,
            'epsilon': epsilon
        })
    
    def _update_loop(self):
        """Background thread for updating plots"""
        clock = pygame.time.Clock()
        
        while self.running:
            try:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        pygame.quit()
                        return
                
                # Update metrics
                updated = False
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get()
                    
                    if metrics['loss'] is not None:
                        self.losses.append(float(metrics['loss']))
                        updated = True
                    if metrics['reward'] is not None:
                        self.avg_rewards.append(float(metrics['reward']))
                        updated = True
                    if metrics['success'] is not None:
                        self.success_rates.append(float(metrics['success']))
                        updated = True
                    if metrics['epsilon'] is not None:
                        self.epsilon_values.append(float(metrics['epsilon']))
                        updated = True
                
                # Update plots if needed
                if updated:
                    self._update_plots()
                
                clock.tick(30)  # Limit to 30 FPS
            except Exception as e:
                print(f"Error in update loop: {e}")
                continue
    
    def _update_plots(self):
        """Update all plot lines and render to pygame surface"""
        try:
            # Update loss plot
            if len(self.losses) > 0:
                self.lines['loss'].set_data(range(len(self.losses)), self.losses)
                self.axs[0, 0].relim()
                self.axs[0, 0].autoscale_view()
            
            # Update reward plot
            if len(self.avg_rewards) > 0:
                self.lines['reward'].set_data(range(len(self.avg_rewards)), self.avg_rewards)
                self.axs[0, 1].relim()
                self.axs[0, 1].autoscale_view()
            
            # Update success rate plot
            if len(self.success_rates) > 0:
                self.lines['success'].set_data(range(len(self.success_rates)), self.success_rates)
                self.axs[1, 0].relim()
                self.axs[1, 0].autoscale_view()
            
            # Update epsilon plot
            if len(self.epsilon_values) > 0:
                self.lines['epsilon'].set_data(range(len(self.epsilon_values)), self.epsilon_values)
                self.axs[1, 1].relim()
                self.axs[1, 1].autoscale_view()
            
            # Draw the plot
            self.canvas.draw()
            
            # Get the RGBA buffer from the figure
            buf = self.canvas.buffer_rgba()
            arr = np.asarray(buf)
            
            # Convert to pygame surface
            width, height = self.canvas.get_width_height()
            surf = pygame.image.frombuffer(arr.tobytes(), (width, height), 'RGBA')
            
            # Display the plot
            scaled_surf = pygame.transform.scale(surf, self.plot_size)
            self.screen.blit(scaled_surf, (0, 0))
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def close(self):
        """Cleanup method"""
        self.running = False
        pygame.quit()
        plt.close(self.fig)
    

# Rest of the code remains the same...

class EnhancedTrainingVisualizer(TrainingVisualizer):
    def __init__(self, env, agent, window_size=200, cell_size=16):
        self.metrics_viz = MetricsVisualizer()
        super().__init__(env, agent, window_size, cell_size)
        
    def update(self):
        if pyxel.btnp(pyxel.KEY_S):
            self.agent.save("maze_agent.pth")
            print("Model saved to maze_agent.pth")
            
        if pyxel.btnp(pyxel.KEY_L):
            self.agent.load("maze_agent.pth")
            print("Model loaded from maze_agent.pth")
        if pyxel.btnp(pyxel.KEY_Q):
            self.metrics_viz.close()
            pyxel.quit()
        
        if not self.training_complete:
            # Training loop
            if self.current_state is None or self.done or self.steps >= self.max_steps:
                if self.current_state is not None:
                    # Record metrics
                    self.episode_rewards.append(self.episode_reward)
                    if len(self.episode_rewards) >= 10:
                        avg_reward = np.mean(list(self.episode_rewards)[-10:])
                        self.avg_rewards.append(avg_reward)
                        self.metrics_viz.update_metrics(
                            reward=avg_reward,
                            success=1.0 if self.done else 0.0,
                            epsilon=self.agent.epsilon
                        )
                    self.success_rate.append(1.0 if self.done else 0.0)
                
                self.episodes_completed += 1
                if self.episodes_completed >= self.num_episodes:
                    self.training_complete = True
                    return
                
                self.reset_episode()
                return
            
            # Take a step in the environment
            action = self.agent.select_action(self.current_state)
            next_state, reward, self.done = self.agent.step(action)
            
            # Store transition and learn
            self.agent.memory.push(self.current_state, action, reward, next_state, self.done)
            loss = self.agent.learn()
            
            if loss is not None:
                self.losses.append(loss)
                self.current_loss = loss
                self.metrics_viz.update_metrics(loss=loss)
            
            self.current_state = next_state
            self.episode_reward += reward
            self.steps += 1

def train_and_visualize(env, agent, num_episodes=1000):
    visualizer = EnhancedTrainingVisualizer(env, agent)
    visualizer.num_episodes = num_episodes


env = Environment(maze_map=maze_map)
agent = DQLAgent(env=env, state_size=103)
train_and_visualize(env=env, agent=agent, num_episodes=10)
visualize_agent(env, agent)
agent.save('maze_agent.pth')
