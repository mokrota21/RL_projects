import numpy as np
import pyxel
from environment import EMPTY_M, WALL_M, SUBGOAL_M, GOAL_M, UNOBSERVED_M, AGENT_M, Agent, Environment, Point
from rl import EPolicy, ValueAction
from collections import deque
from agents import AgentVision
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

ENV = Environment()
RANDOM = True
ENV = Environment(map=maze_map, subgoal_pos=Point(5, 3), goal_pos=Point(8, 7)) # homework
# RANDOM = False

class Renderer:
    def __init__(self, agent: Agent, window_size=200, cell_size=40, num_episodes=1000, max_steps=100):
        self.agent = agent
        self.cell_size = cell_size
        self.window_size = window_size
        
        # Calculate window dimensions
        self.maze_width = self.agent.observation_map.shape[1] * cell_size
        self.maze_height = self.agent.observation_map.shape[0] * cell_size
        
        # Initialize Pyxel with doubled width for split screen
        pyxel.init(
            self.maze_width * 2,  # Double the width for split screen 
            self.maze_height,
            fps=10,
            title="Maze Agent Training Visualization"
        )
        
        # Color mappings for observation map
        self.colors = {
            EMPTY_M: 7,
            WALL_M: 0,
            SUBGOAL_M: 9,
            GOAL_M: 11,
            UNOBSERVED_M: 5,
            AGENT_M: 8
        }
        
        # Start Pyxel
        print('start')
        self.count = 0 
        pyxel.run(self.update, self.draw)
        
    def reset_episode(self):
        ENV.reset(agents=[self.agent], random=RANDOM)
    
    def update(self):
        if pyxel.btnp(pyxel.KEY_R):
            self.reset_episode()    
        ENV.update(hit_wall=False)
        self.count += 1
        if self.count % 50 == 0:
             print(self.agent.reward_history[-50:])
             self.count = 0
    
    def draw_reward_map(self, offset_x):
        # Get reward map from ENV
        if ENV.agents[0].has_subgoal:
            reward_map = ENV.reward_map_goal
        else:
            reward_map = ENV.reward_map_subgoal
        
        # Find min and max rewards for normalization
        min_reward = reward_map.min()
        max_reward = reward_map.max()
        reward_range = max_reward - min_reward if max_reward != min_reward else 1
        
        # Draw reward map
        for y in range(reward_map.shape[0]):
            for x in range(reward_map.shape[1]):
                reward = reward_map[y, x]
                
                # Normalize reward to 0-1 range and convert to color
                # Using color gradient from dark blue (1) to yellow (10)
                normalized_reward = (reward - min_reward) / reward_range
                color = int(1 + normalized_reward * 9)  # Maps to colors 1-10
                
                pyxel.rect(
                    offset_x + x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                    color
                )

                # Draw numerical value
                # Format reward value to 2 decimal places
                value_str = f"{reward:.2f}"
                
                # Calculate text position to center it in the tile
                # Assuming each character is roughly 4 pixels wide and 6 pixels tall
                text_width = len(value_str) * 4
                text_x = offset_x + x * self.cell_size + (self.cell_size - text_width) // 2
                text_y = y * self.cell_size + (self.cell_size - 6) // 2
                
                # Choose text color based on background brightness
                # Use white text for darker backgrounds, black for lighter ones
                text_color = 7 if color < 6 else 0
                
                # Draw the text
                pyxel.text(text_x, text_y, value_str, text_color)
        
    def draw(self):
        pyxel.cls(0)
        
        # Draw observation map on the left side
        any_agent = False
        for y in range(self.agent.observation_map.shape[0]):
            for x in range(self.agent.observation_map.shape[1]):
                cell_value = self.agent.observation_map[y, x]
                if cell_value == AGENT_M:
                     any_agent = True
                color = self.colors.get(cell_value, 0)
                
                pyxel.rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                    color
                )
                
        # Draw reward map on the right side
        self.draw_reward_map(offset_x=self.maze_width)
        
        if not any_agent:
             pass



value_action = ValueAction(500, architecture='one_hidden')
value_action.load('full_vision_long_run.pth')
policy = EPolicy(value_action=value_action, initial_epsilon=0.0)
agent = AgentVision(policy=policy, visibility=100)
ENV.reset([agent], random=RANDOM)
agent.reset(environment=ENV, random=True)
Renderer(agent=agent)