import numpy as np
from random import uniform

class State:
    pass
class Action:
    pass
class Policy:
    pass
class Reward:
    pass
class Value:
    pass

class State:
    def __init__(self, x, y) -> None:
        self.x, self.y = x, y

class Action:
    def __init__(self, state1, state2) -> None:
        self.state1 = state1
        self.state2 = state2

class Reward:
    def __init__(self, reward_m) -> None:
        self.reward_m = reward_m
    
    def compute(self, action: Action):
        destination: State = action.state2
        x, y = destination.x, destination.y
        return self.reward_m[y][x]

def sample_list(list):
    prob = [x[1] for x in list]
    rand_num = uniform(0, 1)
    choice = 0
    while rand_num > 0:
        rand_num -= prob[choice]
    return list[choice][0]

#0 - down, 1 - right, 2 - up, 3 - left
def maze_policy_to_action(z, y, x):
    state1 = State(x, y)
    state2 = State(x, y)

    if z == 0:
        state2.y += 1
    elif z == 1:
        state2.x += 1
    elif z == 2:
        state2.y -= 1
    elif z == 3:
        state2.x -= 1
    else:
        raise("Not maze-like policy")
    
    return Action(state1, state2)

def in_shape(state, shape_x, shape_y):
    return 0 <= state.x < shape_x and 0 <= state.y < shape_y

class Policy:
    def __init__(self, policy_m, policy_to_action) -> None:
        self.policy_m = policy_m
        self.policy_to_action =  policy_to_action
    
    def action(self, state: State):
        sample_states = []
        x = state.x
        y = state.y
        for z in range(self.policy_m.shape[0]):
            sample_states.append((self.policy_to_action(z, x, y), self.policy_m[z, y, x]))

        return sample_list(sample_states)

    def greedy_update(self, value: Value):
        new_policy = np.zeros(self.policy_m.shape)
        for x in range(self.policy_m.shape[2]):
            for y in range(self.policy_m.shape[1]):
                best = -1000
                
                for z in range(self.policy_m.shape[0]):
                    action = self.policy_to_action(z, y, x)
                    state_from = action.state1
                    state_to = action.state2

                    if in_shape(state=state_to, shape_x=self.policy_m.shape[2], shape_y=self.policy_m.shape[1]):
                        v = value.value_m[state_to.y, state_to.x]
                    else:
                        v = -1000
                    
                    best = max(best, v)
                
                amount = 0
                # print(best)
                
                for z in range(self.policy_m.shape[0]):
                    action = self.policy_to_action(z, y, x)
                    state_from = action.state1
                    state_to = action.state2

                    if in_shape(state=state_to, shape_x=self.policy_m.shape[2], shape_y=self.policy_m.shape[1]):
                        # print(x, y, z, value.value_m[state_to.y, state_to.x])
                        if -0.00001 <= value.value_m[state_to.y, state_to.x] - best <= 0.000001:
                            new_policy[z, y, x] = 1
                            amount += 1
                
                new_policy[:, y, x] = new_policy[:, y, x] * float(1 / amount)
        self.policy_m = new_policy


# class Policy:
#     def __init__(self, policy_m, policy_to_action) -> None:
#         self.policy_m = policy_m
#         self.policy_to_action = policy_to_action

#     def action(self, state):
#         sample = []
#         for i in range(self.policy_m.shape[0]):
#             sample.append(i, (self.policy_m[i][state.y][state.x]))
#         sampled_action = sample_list(sample)
#         return self.policy_to_action(sampled_action[0], state.y, state.x)
    
#     # def greedy_update(self, value):
#     #     for x in range(self.policy_m.shape[2]):
#     #         for y in range(self.policy_m.shape[1]):
#     #             best = -1000
#     #             sample = []
                
#     #             for z in range(self.policy_m.shape[0]):
#     #                 action = self.policy_to_action(z, y, x)
#     #                 state_to = action.state2
#     #                 if (0 <= state_to.y < self.policy_m.shape[0] and 0 <= state_to.x < self.policy_m.shape[1]):
#     #                     best = max(best, value.value_m[state_to.y][state_to.x])
#     #             for z in range(self.policy_m.shape[0]):
#     #                 action = self.policy_to_action(z, y, x)
#     #                 state_to = action.state2
#     #                 if (0 <= state_to.y < self.policy_m.shape[0] and 0 <= state_to.x < self.policy_m.shape[1]):
#     #                     if value.value_m[state_to.y][state_to.x] - best <= 0.000001:
#     #                         sample.append((z, y, x))
                
#     #             for pos in sample:
#     #                 self.policy_m[pos] = 1 / len(sample)
                
class Value:
    def __init__(self, value_m, dim=1.0) -> None:
        self.value_m = value_m
        self.dim = dim

    def update(self, policy: Policy, reward: Reward):
        new_value_m = np.zeros(self.value_m.shape)

        for x in range(policy.policy_m.shape[2]):
            for y in range(policy.policy_m.shape[1]):
                for z in range(policy.policy_m.shape[0]):
                    action: Action = policy.policy_to_action(z, y, x)
                    state_to = action.state2
                    if in_shape(state=state_to, shape_x=policy.policy_m.shape[2], shape_y=policy.policy_m.shape[1]):
                        reward_score = reward.compute(action)

                        new_value_m[(y, x)] += policy.policy_m[(z, y, x)] * (reward_score + self.dim * self.value_m[(state_to.y, state_to.x)])
        self.value_m = new_value_m
        return self.value_m

