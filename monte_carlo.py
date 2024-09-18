import numpy as np

class Policy:
    pass

class ValueAction:
    pass

class Environment:
    pass

# States are used in 
class MazeState:
    def __init__(self, y, x) -> None:
        self.yx = (y, x)

    def __eq__(self, other) -> bool:
        if isinstance(other, MazeState):
            return self.yx == other.yx
        return False
    
    def __hash__(self) -> int:
        return hash(self.yx)

# Actions just do what needed, if not inside maze should be checked in environment
class MazeAction:
    def __init__(self, name) -> None:
        pass

    def do_action(self, state):
        return state

class Left(MazeAction):
    def do_action(self, state: MazeState):
        y, x = state.yx
        y -= 1
        return MazeState(y, x)

class Right(MazeAction):
    def do_action(self, state: MazeState):
        y, x = state.yx
        y += 1
        return MazeState(y, x)

class Up(MazeAction):
    def do_action(self, state):
        y, x = state.yx
        x -= 1
        return MazeState(y, x)

class Down(MazeAction):
    def do_action(self, state):
        y, x = state.yx
        x += 1
        return MazeState(y, x)

class MazeReward:
    def __init__(self) -> None:
        pass

    def reward(self, state: MazeState, action: MazeAction):
        n_state = action.do_action(state)

class Environment:
    def __init__(self, reward_map) -> None:
        self.reward_map = reward_map #  function that takes state and action and returns reward. You need to consider cases with unexpected states
    
    def random_state(self):
        pass

from random import randint

class EnvironmentMaze(Environment):
    def __init__(self, reward_map) -> None:
        super().__init__(reward_map)
    
    def random_state(self):
        y_max, x_max = self.reward_map.shape
        
        y = randint(0, y_max - 1)
        x = randint(0, x_max - 1)

        return MazeState(y, x)

# class EnvironmentMaze:
#     def __init__(self, reward_map) -> None:
#         self.reward_map = reward_map

#     def valid_state(self, state: MazeState):
#         y, x = state.yx
#         y_max, x_max = self.reward_map.shape
#         return 0 <= y < y_max and 0 <= x < x_max
    
#     def reward(self, state: MazeState, action: MazeAction):
#         n_state = action.do_action(state)
#         if self.valid_state(n_state):
#             return self.reward_map[state.yx]
#         else:
#             return -1

class ValueAction:
    def __init__(self, initial) -> None:
        self.value_action = initial
    
    def 

class MonteCarloModel:
    def __init__(self, policy: Policy, value_action: ValueAction, env: Environment) -> None:
        self.policy = policy
        self.value_action = value_action
        self.env = env
        
