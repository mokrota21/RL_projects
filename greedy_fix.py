import numpy as np

class Action:
    pass
class Reward:
    pass
class State:
    pass
class Value:
    pass
class Policy:
    pass
class ValueAction:
    pass

class Action:
    def __init__(self, state_from, state_to) -> None:
        self.state_from = state_from
        self.state_to = state_to
    
class Reward:
    def __init__(self, matrix):
        self.m = matrix

def maze_valid_move(x, y, x_to, y_to, n, m):
    neighbour_move = (abs(x - x_to) + abs(y - y_to) == 1)
    in_border_move = 0 <= x_to < n and 0 <= y_to < m

    return neighbour_move and in_border_move

class State:
    def __init__(self, x, y) -> None:
        self.x, self.y, self.n, self.m = x, y

class Value:
    def __init__(self, matrix) -> None:
        self.m = matrix

    def update(self, policy):
        pass

from random import choice

def valid_moves(n, m, x, y):
    moves = []
    for y_to in range(-1, 1):
        for x_to in range(-1, 1):
            if maze_valid_move(x, y, x_to, y_to, n, m):
                moves.append((x_to, y_to))
    return moves

def generate_random_policy(n, m):
    policy = np.empty((m, n), dtype=tuple)
    for y in range(m):
        for x in range(n):
            policy[y, x] = choice(valid_moves(n, m, x, y))
    return policy
    

class Policy:
    def __init__(self, matrix) -> None:
        self.m = matrix
    
    def next_action(self, state: State):
        return self.m[state.y, state.x]
    
class ValueAction:
    def construct(self, value: Value, policy: Policy):
        

