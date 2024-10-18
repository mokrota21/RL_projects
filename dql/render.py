import numpy as np
import pyxel
from environment import EMPTY_M, WALL_M, SUBGOAL_M, GOAL_M, UNOBSERVED_M, AGENT_M
from collections import deque

class TrainingVisualizer:
    def __init__(self, map, window_size=200, cell_size=16):
        self.map = map
        self.cell_size = cell_size
        self.window_size = window_size
        
        # Calculate window dimensions
        self.maze_width = self.map.shape[1] * cell_size
        self.maze_height = self.map.shape[0] * cell_size
        self.metrics_height = 120  # Height for metrics visualization
        
        # Initialize Pyxel with larger window
        pyxel.init(
            max(self.maze_width, self.window_size), 
            self.maze_height + self.metrics_height, 
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
        return
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
                self.colors[AGENT_M]
            )
            
            # Draw subgoal status
            status_color = 11 if self.current_state.has_subgoal else 8
            pyxel.rect(0, 0, 4, 4, status_color)

if __name__ == "__main__":
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
    TrainingVisualizer(maze_map)
