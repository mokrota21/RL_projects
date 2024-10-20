from rl import ValueAction, EPolicy
from environment import Agent, Environment, WALL_M, SUBGOAL_M, GOAL_M, UNOBSERVED_M, EMPTY_M, AGENT_M
from typing import List
import numpy as np
import torch.nn as nn
from metrics import MetricsVisualizer
import pyxel
import numpy as np
from typing import Optional
import pyxel
import numpy as np
from typing import Optional, List, Deque
from collections import deque

# Maze Parameters
STATE_SIZE = 100
#hyper parameters
EPSILON = 0.6
EPSILON_DECAY = 0.9999
HIDDEN_SIZE = 128
BATCH_SIZE = 32
BUFFER_SIZE = 100
DIM = 1.0
ALPHA = 0.0001
NUM_EPISODES = 1000
MAX_STEPS = 100

class TrainingVisualizer:
    def __init__(self, env, value_action, window_size=200, cell_size=16):
        self.env = env
        self.value_action = value_action
        self.policy = EPolicy(value_action=self.value_action, initial_epsilon=EPSILON, decay=EPSILON_DECAY)
        self.agent = Agent(self.policy)
        self.agent.reset(env)
        self.env.reset(agents=[self.agent])
        self.cell_size = cell_size
        self.window_size = window_size
        
        # Calculate window dimensions
        self.maze_width = self.agent.observation_map.shape[1] * cell_size
        self.maze_height = self.agent.observation_map.shape[0] * cell_size
        self.metrics_height = 120  # Height for metrics visualization
        
        # Initialize Pyxel with larger window
        pyxel.init(
            self.maze_width, 
            self.maze_height + self.metrics_height, 
            title="Maze Agent Training Visualization"
        )
        
        # Color mappings
        self.colors = {
            EMPTY_M: 7,      # White
            WALL_M: 0,       # Black
            SUBGOAL_M: 9,    # Orange
            GOAL_M: 11,      # Yellow
            UNOBSERVED_M: 5, # Dark blue
            AGENT_M: 8       # Red
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
            for y in range(self.agent.observation_map.shape[0]):
                for x in range(self.agent.observation_map.shape[1]):
                    cell_value = self.agent.observation_map[y, x]
                    color = self.colors.get(cell_value, 0)
                    
                    pyxel.rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                        color
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
    def __init__(self, window_size=1000):
        self.window_size = window_size
        
        # Metrics storage
        self.losses = deque(maxlen=window_size)
        self.avg_rewards = deque(maxlen=window_size)
        self.goals = deque(maxlen=window_size)
        self.subgoals = deque(maxlen=window_size)
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
        self.lines['goal'], = self.axs[1, 0].plot([], [], 'b-', label='Goal')
        self.lines['subgoal'], = self.axs[1, 0].plot([], [], 'b-', label='Subgoal')
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
    
    def update_metrics(self, loss=None, reward=None, goal=None, subgoal=None, epsilon=None):
        """Thread-safe method to update metrics"""
        self.metrics_queue.put({
            'loss': loss,
            'reward': reward,
            'goal': goal,
            'subgoal': subgoal,
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
                    if metrics['goal'] is not None:
                        self.goals.append(float(metrics['goal']))
                        updated = True
                    if metrics['subgoal'] is not None:
                        self.subgoals.append(float(metrics['subgoal']))
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
            if len(self.goals) > 0:
                self.lines['goal'].set_data(range(len(self.goals)), self.goals)
                self.axs[1, 0].relim()
                self.axs[1, 0].autoscale_view()
            
            if len(self.subgoals) > 0:
                self.lines['subgoal'].set_data(range(len(self.subgoals)), self.subgoals)
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
    def __init__(self, env, value_action, window_size=200, cell_size=16, num_episodes=1000, max_steps=100, print_every=1):
        self.metrics_viz = MetricsVisualizer()
        self.num_episodes = 1000
        self.max_steps = 100
        self.print_every = 1
        super().__init__(env, value_action, window_size, cell_size)
        
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

        policy = self.policy
        agent = self.agent
        env: Environment = self.env
        reward_total = 0
        reward_count = 0
        for episode in range(self.num_episodes):
            print('-' * 100)
            agent.reset(env)
            env.reset(agents=[agent])
            print(env.map)

            current_state = agent.get_state()
            episode_reward = 0

            for _ in range(self.max_steps):
                # RL
                updated = env.update()
                action = agent.action_history[-1]
                next_state = agent.get_state()
                reward = agent.reward_history[-1]
                value_action.memory.push(state=current_state, action=action, reward=reward, next_state=next_state)

                loss = value_action.learn()
                policy.decay_epsilon()

                current_state = next_state
                reward_total += reward
                reward_count += 1
                
                # Visualisation
                self.metrics_viz.update_metrics(loss=loss, reward=reward_total / reward_count, goal=agent.has_goal, subgoal=agent.has_subgoal, epsilon=policy.epsilon)

                if not updated:
                    break
            
            if (episode + 1) % self.print_every == 0:
                print(f"Episode {episode + 1}, Average Reward: {reward_total / reward_count:.2f}")

def train_and_visualize(env, value_action, num_episodes=1000):
    visualizer = EnhancedTrainingVisualizer(env, value_action, num_episodes=num_episodes)


value_action = ValueAction(state_size=STATE_SIZE, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, dim=DIM, alpha=ALPHA)
value_action.load('best_model.pth')

maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
replace_1 = lambda x: WALL_M if x == 1 else x
maze_map = list(map(lambda x: list(map(replace_1, x)), maze_map))
maze_map = np.array(maze_map, dtype=np.int64)

env = Environment(map=maze_map)
train_and_visualize(env=env, value_action=value_action, num_episodes=10)