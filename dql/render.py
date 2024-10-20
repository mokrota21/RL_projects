import numpy as np
import pyxel
from environment import EMPTY_M, WALL_M, SUBGOAL_M, GOAL_M, UNOBSERVED_M, AGENT_M, Agent, Environment
from rl import EPolicy, ValueAction
from collections import deque
import threading

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

ENV = Environment(map=maze_map)

class Renderer:
    def __init__(self, agent: Agent, window_size=200, cell_size=16, num_episodes=1000, max_steps=100):
        self.agent = agent
        self.cell_size = cell_size
        self.window_size = window_size
        
        # Calculate window dimensions
        self.maze_width = self.agent.observation_map.shape[1] * cell_size
        self.maze_height = self.agent.observation_map.shape[0] * cell_size
        
        # Initialize Pyxel with larger window
        pyxel.init(
            self.maze_width, 
            self.maze_height,
            title="Maze Agent Training Visualization"
        )
        
        # Color mappings
        self.colors = {
            EMPTY_M: 7,
            WALL_M: 0,
            SUBGOAL_M: 9,
            GOAL_M: 11,
            UNOBSERVED_M: 5,
            AGENT_M: 8
        }
        
        # Episode tracking
        self.current_episode = 0
        self.episodes_completed = 0
        self.goal = False
        self.steps = 0
        self.max_steps = max_steps
        self.episode_reward = 0
        self.current_loss = 0
        self.training_complete = False
        
        # Training parameters
        self.num_episodes = num_episodes
        self.print_every = 10
        
        # Start Pyxel
        print('start')
        pyxel.run(self.update, self.draw)
    
    def reset_episode(self):
        self.current_state = self.agent
        self.goal = False
        self.steps = 0
        self.episode_reward = 0
    
    def update(self):
        ENV.update(hit_wall=True)
        
    def draw(self):
        pyxel.cls(0)
        
        # Draw maze
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
        
        metrics_y = self.maze_height

        # Draw subgoal status
        status_color = 11 if self.agent.has_subgoal else 8
        pyxel.rect(0, 0, 4, 4, status_color)
        
        progress = f"Episode: {self.episodes_completed}/{self.num_episodes}"
        pyxel.text(10, metrics_y - 10, progress, 7)
        
        # Draw current episode reward
        reward_text = f"Current Reward: {self.episode_reward:.1f}"
        pyxel.text(10, metrics_y - 20, reward_text, 7)
        
        # Draw current loss
        loss_text = f"Current Loss: {self.current_loss:.3f}"
        pyxel.text(10, metrics_y - 30, loss_text, 7)
        
        # Draw current step
        step_text = f"Current Step: {self.steps}"
        pyxel.text(10, metrics_y - 40, step_text, 7)



value_action = ValueAction(100, 128)
value_action.load('best_model.pth')
policy = EPolicy(value_action=value_action, initial_epsilon=0.0)
agent = Agent(policy=policy)
agent.reset(environment=ENV)
ENV.reset([agent])
Renderer(agent=agent)