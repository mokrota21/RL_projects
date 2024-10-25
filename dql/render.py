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

# ENV = Environment(map=maze_map)
ENV = Environment(map=maze_map, subgoal_pos=Point(5, 3), goal_pos=Point(8, 7)) # homework

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
            fps=10,
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
        
        # Start Pyxel
        print('start')
        pyxel.run(self.update, self.draw)
    
    def reset_episode(self):
        ENV.reset(agents=[self.agent], random=False)
    
    def update(self):
        if pyxel.btnp(pyxel.KEY_R):
            self.reset_episode()    
        ENV.update(hit_wall=True)
        
    def draw(self):
        pyxel.cls(0)
        
        # Draw maze
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
        if not any_agent:
             pass



value_action = ValueAction(500, architecture='small_one_hidden')
# value_action.load('nlearning.pth')
policy = EPolicy(value_action=value_action, initial_epsilon=0.0)
agent = AgentVision(policy=policy, visibility=100)
agent.reset(environment=ENV)
ENV.reset([agent], random=False)
Renderer(agent=agent)