from base import State
from collections import deque
import numpy as np
from random import choice, sample
from time import sleep
from typing import List

### Env mapping
EMPTY_M = 0
SUBGOAL_M = 1
GOAL_M = 2
WALL_M = 3
UNOBSERVED_M = 4
AGENT_M = 5
### Env reward
EMPTY_R = -0.1
WALL_R = -1.0
SUBGOAL_R = 5.0
GOAL_R = 10.0
### Agent params
VISIBILITY = 2

class Point:
    pass
class Agent:
    pass
class Environment:
    pass
class Policy:
    def next_action(self, *args, **kwargs):
        pass

class Point:
    def __init__(self, y, x):
        self.yx = (y, x)
    
    def __call__(self, *args, **kwds):
        return self.yx

    def __add__(self, other: Point):
        if isinstance(other, Point):
            y1, x1 = self.yx
            y2, x2 = other.yx
            return Point(y1 + y2, x1 + x2)

    def __sub__(self, other):
            y1, x1 = self.yx
            y2, x2 = other.yx
            return Point(y1 - y2, x1 - x2)


    def __str__(self):
        return str(self.yx)
    
    __repr__ = __str__

### Actions
ACTIONS = [
    Point(-1, 0), # up
    Point(1, 0), # down
    Point(0, 1), # right
    Point(0, -1) # left
]
UP = ACTIONS[0]
DOWN = ACTIONS[1]
RIGHT = ACTIONS[2]
LEFT = ACTIONS[3]
###

def random_action_index():
        return choice([0, 1, 2, 3])

class Environment:
    def __init__(self, map: np.ndarray = None, kill_range: int = 1, goal_pos = None, subgoal_pos = None) -> None:
        if map is not None:
            self.map = map # We generate current map online without storing since it can be restored by agent history if necessary
        else:
            self.map = self.generate_maze(10, 10)
        if goal_pos is None:
            assert subgoal_pos is None
            self.random_goal_subgoal()
        else:
            self.goal_pos: Point = goal_pos
            self.subgoal_pos: Point = subgoal_pos
        self.draw_goal_subgoal()

        self.agents: List[Agent] = []
        self.kill_range = kill_range

    def generate_maze(self, height, width):
        # Initialize maze with walls
        maze = np.full((height, width), WALL_M, dtype=np.int8)
        
        # Define starting point (must be odd coordinates to ensure paths don't merge)
        start_y, start_x = 1, 1
        maze[start_y, start_x] = EMPTY_M
        
        # Stack for DFS
        stack = [(start_y, start_x)]
        
        # Directions: (y, x) - up, right, down, left
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        
        while stack:
            current_y, current_x = stack[-1]
            
            # Get all valid neighbors that haven't been visited
            valid_neighbors = []
            for dy, dx in directions:
                new_y, new_x = current_y + dy, current_x + dx
                
                # Check if the neighbor is within bounds and not visited
                if (0 < new_y < height-1 and 
                    0 < new_x < width-1 and 
                    maze[new_y, new_x] == WALL_M):
                    valid_neighbors.append((new_y, new_x))
            
            if valid_neighbors:
                # Choose random neighbor
                next_y, next_x = choice(valid_neighbors)
                
                # Carve path by setting the cell between current and neighbor to empty
                maze[current_y + (next_y - current_y)//2, 
                     current_x + (next_x - current_x)//2] = EMPTY_M
                maze[next_y, next_x] = EMPTY_M
                
                stack.append((next_y, next_x))
            else:
                stack.pop()
        
        return maze
    
    def draw_goal_subgoal(self):
        self.map[self.goal_pos.yx] = GOAL_M
        self.map[self.subgoal_pos.yx] = SUBGOAL_M
    
    def remove_goal_subgoal(self):
        self.map[self.goal_pos.yx] = EMPTY_M
        self.map[self.subgoal_pos.yx] = EMPTY_M
    
    def random_goal_subgoal(self):
        self.goal_pos, self.subgoal_pos = self.random_position(2)

    # def get_reward(self, pos: Point, has_subgoal: bool):
    #     tile = self.map[pos.yx]
    #     if tile == WALL_M:
    #         return WALL_R
    #     elif tile == SUBGOAL_M and not has_subgoal:
    #         return SUBGOAL_R
    #     elif tile == GOAL_M and has_subgoal:
    #         return GOAL_R
    #     else:
    #         return EMPTY_R
    
    def valid_actions(self, pos: Point):
        valid_actions = []
        for i in range(len(ACTIONS)):
            action = ACTIONS[i]
            new_pos = pos + action
            if self.pos_inside(new_pos) and self.map[new_pos.yx] != WALL_M:
                valid_actions.append(i)
        return valid_actions

    def pos_inside(self, pos: Point):
        y, x= pos.yx
        return 0 <= y < self.map.shape[0] and 0 <= x < self.map.shape[1]

    def random_position(self, amount=1):
        empty_points = []
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] == EMPTY_M:
                    empty_points.append(Point(y, x))
        return sample(empty_points, amount)
    
    def reset(self, agents: List[Agent], random=True):
        if random:
            self.map = self.generate_maze(10, 10)
            self.remove_goal_subgoal()
            self.random_goal_subgoal()
            self.draw_goal_subgoal()
        self.agents = []
        for agent in agents:
            agent.reset(self)
            self.agents.append(agent)
    
    def update_agents(self, hit_wall):
        updated = 0
        for agent in self.agents:
            if agent.has_goal:
                continue
            agent.update()
            
            old_pos = agent.pos_history[-1]
            action = agent.action_history[-1]
            new_pos = old_pos + ACTIONS[action]
            
            reward = None
            if self.map[new_pos.yx] == WALL_M:
                if hit_wall:
                    print(f"Agent {agent} bumped in the wall: tried to reach position {new_pos} from {old_pos}")
                reward = WALL_R
                new_pos = old_pos
            elif self.map[new_pos.yx] == SUBGOAL_M and not agent.has_subgoal:
                reward = SUBGOAL_R
                agent.has_subgoal = True
                # print('subgoal')
            elif self.map[new_pos.yx] == GOAL_M and agent.has_subgoal:
                reward = GOAL_R
                agent.has_goal = True
                # print('goal')
            else:
                reward = EMPTY_R
            agent.set_reward(reward)
            agent.pos_history.append(new_pos)
            updated += 1

        return updated > 0
    
    def update(self, hit_wall=False):
        "If something doesn't update we terminate whole episode"
        updated = True
        updated = updated and self.update_agents(hit_wall=hit_wall)

        return updated

def maze_observation_encoded(observation_map: np.ndarray):
    rows, columns = observation_map.shape
    tile_type = [EMPTY_M, WALL_M, GOAL_M, UNOBSERVED_M, AGENT_M]
    res = np.zeros((rows, columns, len(tile_type)), dtype=np.int32)
    for i in range(len(tile_type)):
        tile = tile_type[i]
        res[observation_map == tile, i] = 1
    return res

class Agent:
    "Class that provides communication between Policy and Environment"
    def __init__(self, policy: Policy, visibility: int = VISIBILITY) -> None:
        self.policy = policy
        self.visibility = visibility

        # Debug info
        self.pos_history = None
        self.action_history = None
        self.reward_history = None

        # State info
        self.environment = None
        self.observation_map = None
        self.total_reward = None
        self.has_subgoal = None
        self.has_goal = None

    def reset_history(self):
        self.pos_history = deque([self.environment.random_position()[0]])
        self.action_history = deque()
        self.observation_map = np.ones(shape=self.environment.map.shape, dtype=int) * UNOBSERVED_M # for full obs
        # self.observation_map = np.full(shape=(self.visibility * 2 + 1, self.visibility * 2 + 1), fill_value=UNOBSERVED_M, dtype=int) # for partial
        self.reward_history = []
    
    def get_state(self):
        return State(features=[maze_observation_encoded(self.observation_map)])

    def reset(self, environment: Environment):
        self.environment = environment
        self.reset_history()

        self.total_reward = 0
        self.has_goal = False
        self.has_subgoal = False

    def hide_obj(self, pos):
        yx = pos.yx
        tile = int(self.observation_map[yx])
        if tile == AGENT_M:
            return
        elif tile != UNOBSERVED_M:
            self.observation_map[yx] = EMPTY_M
    
    def restore_obj(self, pos):
        yx = pos.yx
        tile = int(self.observation_map[yx])
        if tile == AGENT_M:
            return
        elif tile != UNOBSERVED_M:
            self.observation_map[yx] = GOAL_M

    def update_vision(self):
        "Agent can see only current goal. It means if he doesn't have subgoal he can't see goal, otherwise he can't see subgoal. Both will be represented by the same number"
        # Observe whole map
        map = self.environment.map
        pos = self.pos_history[-1].yx
        range_obs = max(pos[0] - self.visibility, 0), min(pos[0] + self.visibility + 1, map.shape[0]), \
            max(pos[1] - self.visibility, 0), min(pos[1] + self.visibility + 1, map.shape[1])
        self.observation_map[range_obs[0]:range_obs[1], range_obs[2]:range_obs[3]] = map[range_obs[0]:range_obs[1], range_obs[2]:range_obs[3]]
        self.observation_map[pos] = AGENT_M

        goal_pos = self.environment.goal_pos
        subgoal_pos = self.environment.subgoal_pos
        subgoal_inside = True
        goal_inside = True

        # Observe only in Visibility range
        # map = self.environment.map.copy()
        # pos = self.pos_history[-1].yx
        # range_obs = pos[0] - self.visibility, pos[0] + self.visibility + 1, \
        #     pos[1] - self.visibility, pos[1] + self.visibility + 1
        # obs_size = self.visibility * 2 + 1
        # self.observation_map = np.empty((obs_size, obs_size))
        # for y in range(range_obs[0], range_obs[1]):
        #     for x in range(range_obs[2], range_obs[3]):
        #         y_obs, x_obs = y - range_obs[0], x - range_obs[2]
        #         if y < 0 or y >= map.shape[0] or x < 0 or x >= map.shape[1]:
        #             self.observation_map[y_obs, x_obs] = WALL_M
        #         else:
        #             self.observation_map[y_obs, x_obs] = map[y, x]

        # y = self.visibility
        # x = self.visibility
        # left_up_pos = Point(pos[0], pos[1]) - Point(y, x)
        
        # subgoal_pos = self.environment.subgoal_pos - left_up_pos
        # subgoal_inside = 0 <= subgoal_pos.yx[0] < self.observation_map.shape[0]
        # subgoal_inside = subgoal_inside and 0 <= subgoal_pos.yx[1] < self.observation_map.shape[1]

        # goal_pos = self.environment.goal_pos - left_up_pos
        # goal_inside = 0 <= goal_pos.yx[0] < self.observation_map.shape[0]
        # goal_inside = goal_inside and 0 <= goal_pos.yx[1] < self.observation_map.shape[1]

        if self.has_subgoal:
            if subgoal_inside:
                self.hide_obj(subgoal_pos)
            if goal_inside:
                self.restore_obj(goal_pos)
        else:
            if goal_inside:
                self.hide_obj(goal_pos)
            if subgoal_inside:
                self.restore_obj(subgoal_pos)
    
    def update(self):
        "Always updates history even if invalid action. If it is invalid revert is called"
        self.update_vision()
        action_index = self.policy.next_action(self) # Only based on what we see. It is responsibility of environment to update history when calling update
        self.action_history.append(action_index)
    
    def set_reward(self, reward: float):
        self.reward_history.append(reward)
        self.total_reward += reward

    def wall_bump(self):
        self.pos_history.pop()
        self.pos_history.append(self.pos_history[-1])
        self.reward_history.append(-10)