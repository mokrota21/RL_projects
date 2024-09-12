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
    in_border_move = (0 <= x_to < n and 0 <= y_to < m)

    return neighbour_move and in_border_move

def maze_possible_actions(state: State):
    x, y = state.xy
    options = []

    options.append(State(x - 1, y, *state.shape, maze_possible_actions))
    options.append(State(x, y + 1, *state.shape, maze_possible_actions))
    options.append(State(x + 1, y, *state.shape, maze_possible_actions))
    options.append(State(x, y - 1, *state.shape, maze_possible_actions))

    res = []
    for option in options:
        if maze_valid_move(*state.xy, *option.xy, *state.shape):
            res.append(Action(state, option))
        else:
            res.append(None)
    return res

class State:
    def __init__(self, x, y) -> None:
        self.xy = x, y

class Value:
    def __init__(self, matrix, possible_actions, dim=0.1) -> None:
        self.m = matrix
        self.dim = dim
        self.possible_actions = possible_actions

    def next_states_vector(self, state: State, reward: Reward):
        next_actions = self.possible_actions(state)
        res = np.zeros((len(next_actions), 1))
        for i in range(len(next_actions)):
            action = next_actions[i]
            if action is not None:
                res[i] = self.m[action.state_to.xy] * self.dim + reward.m[action.state_to.xy]
        return res

    def update_state(self, policy: Policy, reward: Reward, state: State):
        old_v = self.m[state.xy]
        x, y = state.xy
        value_vector = self.next_states_vector(state)
        policy_vector = policy.m[:, y, x]
        new_v = np.dot(policy_vector, value_vector)
        self.m[state.xy] = new_v
        return abs(old_v - new_v)
    
    def update(self, policy: Policy, reward: Reward, theta=0.001):
        c_theta = 0
        while True:
            for y in range(self.m.shape[0]):
                for x in range(self.m.shape[1]):
                    state = State(x, y, self.m.shape[1], self.m.shape[0], self.possible_actions)
                    c_theta = max(c_theta, self.update_state(policy, reward, state))
            if c_theta < theta:
                return

# from random import choice

# def valid_moves(n, m, x, y):
#     moves = []
#     for y_to in range(-1, 1):
#         for x_to in range(-1, 1):
#             if maze_valid_move(x, y, x_to, y_to, n, m):
#                 moves.append((x_to, y_to))
#     return moves

# def generate_random_policy(n, m):
#     policy = np.empty((m, n), dtype=tuple)
#     for y in range(m):
#         for x in range(n):
#             policy[y, x] = choice(valid_moves(n, m, x, y))
#     return policy
    
# 0 - down, 1 - right, 2 - up, 3 - left
def maze_policy_to_action(z, y, x):
    state
    if z == 0:


class Policy:
    def __init__(self, matrix, policy_to_action) -> None:
        self.m = matrix
        self.policy_to_action = policy_to_action
    
    def next_action(self, state: State):
        return self.m[state.y, state.x]
