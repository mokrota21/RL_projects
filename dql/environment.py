from collections import deque
import numpy as np
from abc import ABC, classmethod
from random import choice
from time import sleep
from typing import List

### Env mapping
EMPTY_M = 0
SUBGOAL_M = -1
GOAL_M = -2
WALL_M = -3
UNOBSERVED_M = -4
AGENT_M = -5
### Env reward
EMPTY_R = -1
WALL_R = -10
SUBGOAL_R = 25
GOAL_R = 100
### Agent params
VISIBILITY = 2

class Point:
    pass
class Agent:
    pass
class Environment:
    pass
class Policy:
    @classmethod
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

    def __str__(self):
        return str(self.yx)
    
    __repr__ = __str__

### Actions
all_actions = [
    Point(-1, 0), # up
    Point(1, 0), # down
    Point(0, 1), # right
    Point(0, -1) # left
]
UP = all_actions[0]
DOWN = all_actions[1]
RIGHT = all_actions[2]
LEFT = all_actions[3]
###

class Environment:
    def __init__(self, reward_map: np.ndarray, kill_range: int = 1) -> None:
        self.agents: List[Agent] = []
        self.reward_map = reward_map # We generate current map online without storing since it can be restored by agent history if necessary
        self.kill_range = kill_range
    
    def valid_actions(self, pos: Point):
        valid_actions = []
        for i in range(len(all_actions)):
            action = all_actions[i]
            new_pos = pos + action
            if self.pos_inside(new_pos) and self.reward_map[new_pos.yx] != WALL_M:
                valid_actions.append(i)
        return valid_actions

    def pos_inside(self, pos: Point):
        y, x= pos.yx
        return 0 <= y < self.reward_map.shape[0] and 0 <= x < self.reward_map.shape[1]

    def random_action(self):
        return choice([0, 1, 2, 3])

    def random_position(self):
        empty_points = []
        for y in range(self.reward_map.shape[0]):
            for x in range(self.reward_map.shape[1]):
                if self.reward_map[y, x] == EMPTY_M:
                    empty_points.append(Point(y, x))
        return choice(empty_points)
    
    def reset(self, agents: List[Agent]):
        self.map = self.reward_map
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
            
            new_pos = agent.pos_history[-1]
            old_pos = agent.pos_history[-2]
            
            reward = None
            if self.reward_map[new_pos.yx] == WALL_M:
                if hit_wall:
                    print(f"Agent {agent} bumped in the wall: tried to reach position {new_pos} from {old_pos}")
                reward = WALL_R
            elif self.reward_map[new_pos.yx] == SUBGOAL_M and not agent.has_subgoal:
                reward = SUBGOAL_R
            elif self.reward_map[new_pos.yx] == GOAL_M and agent.has_subgoal:
                reward = GOAL_R
            else:
                reward = EMPTY_R
            agent.set_reward(reward)
            updated += 1

        return updated > 0
    
    def update(self, hit_wall=False):
        "If something doesn't update we terminate whole episode"
        updated = True
        updated = updated and self.update_agents(hit_wall=hit_wall)

        return updated

    def play(self, agents, f_before=None, f_after=None, render=None, cooldown: float = 0.0, max_steps: int = None):
        self.reset(agents)
        counter = 0
        while max_steps is None or counter < max_steps:
            if f_before:
                f_before()
            if not self.update():
                break
            if f_after:
                f_after()
            if render:
                render(self)
            sleep(cooldown)
            counter += 1

class Agent:
    "Class that provides communication between Policy and Environment"
    def __init__(self, pos: Point, policy: Policy, visibility: int = VISIBILITY) -> None:
        self.initial_pos = pos
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
        self.pos_history = deque([self.initial_pos])
        self.action_history = deque([-1])
        self.observation_map = np.ones(shape=self.environment.reward_map.shape, dtype=int) * UNOBSERVED_M
        self.reward_history = []

    def reset(self, environment: Environment):
        self.environment = environment
        self.reset_history()

        self.total_reward = 0
        self.has_goal = False
        self.has_subgoal = False

    def update_vision(self):
        map = self.environment.reward_map
        pos = self.pos_history[-1].yx
        range = max(pos[0] - self.visibility, 0), min(pos[0] + self.visibility + 1, map.shape[0]), \
            max(pos[1] - self.visibility, 0), min(pos[1] + self.visibility + 1, map.shape[1])
        self.observation_map[range[0]:range[1], range[2]:range[3]] = map[range[0]:range[1], range[2]:range[3]]
    
    def update(self):
        "Always updates history even if invalid action. If it is invalid revert is called"
        self.update_vision()
        action_index = self.policy.next_action(self) # Only based on what we see, such approach doesn't generalize, it will basically understand structure of maze we gave to it
        self.action_history.append(action_index)
        self.action_history.popleft()
        self.pos_history.append(self.pos_history[-1] + all_actions[action_index])
    
    def set_reward(self, reward: float):
        self.reward_history.append(reward)
        self.total_reward += reward

    def wall_bump(self):
        self.pos_history.pop()
        self.pos_history.append(self.pos_history[-1])
        self.reward_history.append(-10)