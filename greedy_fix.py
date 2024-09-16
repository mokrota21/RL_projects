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

def maze_possible_actions(state: State, n, m):
    x, y = state.xy
    options = []

    options.append(State(x, y + 1)) # Down
    options.append(State(x + 1, y)) # Right
    options.append(State(x, y - 1)) # Up
    options.append(State(x - 1, y)) # Left

    res = []
    for option in options:
        if maze_valid_move(*state.xy, *option.xy, n, m):
            res.append(Action(state, option))
        else:
            res.append(None)
    return res

class State:
    def __init__(self, x, y) -> None:
        self.xy = x, y

    def __eq__(self, other) -> bool:
        is_state = isinstance(other, self.__class__)

        if is_state:
            return other.xy == self.xy
        else:
            return False
    
    def action(self, action: Action):
        assert action.state_from == self
        self = action.state_to


class Value:
    def __init__(self, matrix, possible_actions=maze_possible_actions, dim=0.1) -> None:
        self.m = matrix
        self.dim = dim
        self.possible_actions = possible_actions
    
    def value_action(self, action: Action, reward: Reward):
        x, y = action.state_to.xy
        x_from, y_from = action.state_from.xy
        r = reward.m[y_from, x_from]
        return r + self.dim * self.m[y, x]
    
    def value(self, policy: Policy, reward: Reward, state: State):
        action = policy.next_action(state)
        return self.value_action(action, reward)

    def next_states_vector(self, state: State, reward: Reward):
        next_actions = self.possible_actions(state, self.m.shape[1], self.m.shape[0])
        res = np.zeros((len(next_actions), 1))
        for i in range(len(next_actions)):
            action = next_actions[i]
            if action is not None:
                x, y = action.state_to.xy
                x_from, y_from = action.state_from.xy
                res[i] = self.m[y, x] * self.dim + reward.m[y_from,  x_from]
        return res

    def update_state(self, policy: Policy, reward: Reward, state: State):
        x, y = state.xy
        old_v = self.m[y, x]
        value_vector = self.next_states_vector(state, reward=reward)
        policy_vector = policy.m[:, y, x]
        new_v = np.dot(policy_vector, value_vector)
        self.m[y, x] = new_v
        return abs(old_v - new_v)
    
    def update(self, policy: Policy, reward: Reward, theta=0.001):
        while True:
            c_theta = 0
            # print(self.m)
            for y in range(self.m.shape[0]):
                for x in range(self.m.shape[1]):
                    state = State(x, y)
                    c_theta = max(c_theta, self.update_state(policy, reward, state))
            # print(self.m)
            if c_theta < theta:
                return
            # else:
                # print(c_theta)

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
    state_from = State(x, y)
    x_to, y_to = x, y
    if z == 0:
        y_to += 1
    elif z == 1:
        x_to += 1
    elif z == 2:
        y_to -= 1
    elif z == 3:
        x_to -= 1
    else:
        raise(f"Error, unexpected value z={z} in policy to action")
    state_to = State(x_to, y_to)
    return Action(state_from=state_from, state_to=state_to)

class Policy:
    def __init__(self, matrix, policy_to_action=maze_policy_to_action) -> None:
        self.m = matrix
        self.policy_to_action = policy_to_action
    
    def next_action(self, state: State):
        x, y = state.xy
        for z in range(self.m.shape[0]):
            if self.m[z, y, x] == 1.0:
                return self.policy_to_action(z, y, x)
        raise(f"No action from state {state.xy} in policy {self}")

    def update(self, value: Value, reward: Reward):
        count = 0
        for y in range(self.m.shape[1]):
            for x in range(self.m.shape[2]):
                state_from = State(x, y)
                best_move = (-1000000, 0)
                for z in range(self.m.shape[0]):
                    new_action = self.policy_to_action(z, y, x)
                    if maze_valid_move(*new_action.state_from.xy, *new_action.state_to.xy, self.m.shape[2], self.m.shape[1]):
                        new_value = value.value_action(action=new_action, reward=reward)
                        # print(z, y, x, new_value)

                        if best_move[0] < new_value:
                            best_move = (new_value, z)

                z = best_move[1]
                count += 1 if (z != self.m[:, y, x].argmax()) else 0
                self.m[:, y, x] = 0.0
                self.m[z, y, x] = 1.0
        
        return count

