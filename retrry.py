import numpy as np

### Reward Map tiles meaning
EMPTY = 0
SUBGOAL = -1
GOAL = -2
# AGENTS = n where n is number of agents on the tile

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
    "Absolute position in the maze"
    def __init__(self, y, x) -> None:
        self.yx = (y, x)
    
    def __add__(self, other):
        if isinstance(other, Point):
            sy, sx = self.yx
            oy, ox = other.yx
            return Point(sy + oy, sx + ox)

class State:
    "Flexible state class where we put features"
    def __init__(self, **kwargs) -> None:
        self.features = kwargs

class Action:
    "Action class allows to make action given pos: Point and returns new_pos: Point"
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
    "Action that moves object left along x axis"
    def do_action(self, pos: Point):
        add = Point(0, -1)
        return pos + add

class Right(Action):
    "Action that moves object right along x axis"
    def do_action(self, pos: Point):
        add = Point(0, 1)
        return pos + add

class Up(Action):
    "Action that moves object up along y axis (due to nature of lists and arrays it is actually decrementing)"
    def do_action(self, pos: Point):
        add = Point(-1, 0)
        return pos + add

class Down(Action):
    "Action that moves object down along y axis (due to nature of lists and arrays it is actually incrementing)"
    def do_action(self, pos: Point):
        add = Point(1, 0)
        return pos + add

class Agent:
    "Class that provides communication between Actions and Environment"
    def __init__(self, pos: Point, policy: Policy, role: str = 'prey', visibility: int = 2) -> None:
        self.initial_pos = pos
        self.role = role
        self.policy = policy
        self.visibility = visibility
        
        self.pos = None # to avoid overfitting for this exact maze, we are not including position in features
        self.death = None
        self.has_subgoal = None
        self.has_goal = None
    
    def reset(self):
        self.pos = self.initial_pos
        self.death = False
        self.has_goal = False
        self.has_subgoal = False

    def full_vision(self, environment):
        map = environment.map
        vis_shape = map.shape
        vis_shape = (vis_shape[0] + self.visibility * 2, vis_shape[1] + self.visibility * 2)
        visibility_map = np.zeros(shape=vis_shape, dtype=int)
        visibility_map[self.visibility:-self.visibility, self.visibility:-self.visibility] = map
        return visibility_map
    
    # def action(self):
    #     pass

    def update(self, environment):
        self.vision_map = self.full_vision(environment)


class Environment:
    def __init__(self, reward_map: np.ndarray, kill_range: int = 1) -> None:
        self.agents = []
        self.reward_map = reward_map
        self.kill_range = kill_range

        self.map = None
        self.agents = None
    
    def reset(self, agents):
        self.agents = agents
        self.map = self.reward_map
        for agent in self.agents:
            agent.reset()
    
    def update(self):
        self.update_preys()
        self.update_hunters()
        self.death_update()

    def play(self, agents):
        self.reset(agents)
        while True:
            if not self.update():
                break

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