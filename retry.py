import numpy as np

class State:
    pass
class Action:
    pass
class Environment:
    pass
class Policy:
    pass
class ValueAction:
    pass
class Agent:
    pass

class Point:
    def __init__(self, y, x) -> None:
        self.yx = (y, x)
    
    def __add__(self, other):
        if isinstance(other, Point):
            sy, sx = self.yx
            oy, ox = other.yx
            return Point(sy + oy, sx + ox)

class State:
    def __init__(self, **kwargs) -> None:
        self.features = kwargs

class Action:
    def __new__(cls, name: str):
        if name.lower() == "right":
            return super().__new__(Right)
        elif name.lower() == "left":
            return super().__new__(Left)
        elif name.lower() == "up":
            return super().__new__(Up)
        elif name.lower() == "down":
            return super().__new__(Down)
        return super().__new__(cls)

class Left(Action):
    def do_action(self, pos: Point):
        add = Point(0, -1)
        return pos + add

class Right(Action):
    def do_action(self, pos: Point):
        add = Point(0, 1)
        return pos + add

class Up(Action):
    def do_action(self, pos: Point):
        add = Point(-1, 0)
        return pos + add

class Down(Action):
    def do_action(self, pos: Point):
        add = Point(1, 0)
        return pos + add

class Agent:
    def __init__(self, pos: Point, policy: Policy, role: str = 'prey') -> None:
        self.initial_pos = pos
        self.role = role
        self.policy = policy
        
        self.pos = None
        self.death = None
        self.has_subgoal = None
        self.has_goal = None
    
    def reset(self):
        self.pos = self.initial_pos
        self.death = False
        self.has_goal = False
        self.has_subgoal = False
    
    def action(self):
        self

### Reward Map tiles meaning
EMPTY = 0
SUBGOAL = -1
GOAL = -2
# AGENTS = n where n is number of agents on the tile

class Environment:
    def __init__(self, reward_map: np.ndarray, kill_range: int = 1) -> None:
        self.agents = []
        self.reward_map = reward_map
        self.kill_range = kill_range
    
    def update(self):
        self.update_preys()
        self.update_hunters()
        self.death_update()


    def play(self, agents):
        self.agents = agents
        while True:
            self.update()

class ValueAction:
    def __init__(self, initial: dict = {}, default: float = 0) -> None:
        self.value_action = initial
        self.default = default
    
    def value(self, state: State, action: Action):
        return self.value_action.get((state, action), self.default)
    
    # Fix this
    def max_value(self, state: State):
        pass

from random import uniform

class Policy:
    def __init__(self, initial: dict = {}, epsilon: float = 0.01) -> None:
        self.policy = initial
        self.epsilon = epsilon
    
    def next_action(self, state: State, env: Environment) -> Action:
        if uniform(0, 1) < self.epsilon:
            return env.random_action(state)
        return self.policy.get(state, env.random_action(state))