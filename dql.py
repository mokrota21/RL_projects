import numpy as np
from random import uniform, choice
import torch
import torch.nn as nn
import random
import numpy as np
from time import sleep

### Reward Map tiles meaning
EMPTY = 0
SUBGOAL = -1
GOAL = -2
WALL = -3
###

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

MEMORY = 1 # how many elements from history agent remembers
VISIBILITY = 2 # how many tiles around itself agent can see
input_size = (VISIBILITY * 2 + 1) ** 2 * MEMORY + MEMORY + 1
output_size = 1

maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, SUBGOAL, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, GOAL, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
replace_1 = lambda x: WALL if x == 1 else x
maze_map = list(map(lambda x: list(map(replace_1, x)), maze_map))
maze_map = np.array(maze_map)

class Point:
    pass
class State:
    pass
class Agent:
    pass
class Environment:
    pass
class ValueAction:
    pass
class DQLModel:
    pass
class Policy:
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

def numpy_to_list(ar):
    res = []
    for row in ar.tolist():
        res += row
    return res

class State:
    "Put all features as a list"
    def __init__(self, features: tuple):
        self.features = []
        for feature in features:
            try:
                feature = list(map(lambda x: numpy_to_list(x), feature))
                tmp = []
                for f in feature:
                    tmp += f
                feature = tmp
            except:
                pass
            self.features += feature
        print(len(self.features), features)


all_actions = [
    Point(-1, 0), # up
    Point(1, 0), # down
    Point(0, 1), # right
    Point(0, -1) # left
]


class Agent:
    "Class that provides communication between Actions and Environment"
    def __init__(self, pos: Point, policy: Policy, role: str = 'prey', visibility: int = VISIBILITY) -> None:
        self.initial_pos = pos
        self.role = role
        self.policy = policy
        self.visibility = visibility
        
        self.pos_history = None
        self.vision_history = None
        self.action_history = None
        self.reward_history = None
        self.death = None
        self.has_subgoal = None
        self.has_goal = None
        self.got_subgoal = -1
        self.got_goal = -1
    
    def reset(self, environment):
        self.pos_history = [self.initial_pos]

        cur_vision = self.full_vision(environment=environment)
        empty_vision = np.ones(cur_vision.shape) * 100 # non-existent vision to fill the rest of memory
        self.vision_history = [empty_vision * (MEMORY - 1), cur_vision]

        self.action_history = []
        empty_action = -1 # non-existent action
        self.action_history = [empty_action * (MEMORY - 1)]

        self.reward_history = []
        self.death = False
        self.has_goal = False
        self.has_subgoal = False
        self.got_subgoal = -1
        self.got_goal = -1

    def full_vision(self, environment):
        map = environment.reward_map
        vis_shape = map.shape
        vis_shape = (vis_shape[0] + self.visibility * 2, vis_shape[1] + self.visibility * 2)
        visibility_map = np.ones(shape=vis_shape, dtype=int) * WALL
        visibility_map[self.visibility:-self.visibility, self.visibility:-self.visibility] = map
        pos = self.pos_history[-1].yx
        visibility_map = visibility_map[pos[0]:2 * self.visibility + pos[0] + 1, pos[1]: 2 * self.visibility + pos[1] + 1]
        return visibility_map
    
    def get_state(self, to: int = None):
        return State((self.vision_history[-MEMORY:to], self.action_history[-MEMORY:to], [int(self.has_subgoal and self.got_subgoal <= to)])) # this way agent remembers actions it's already performed
    
    def update(self, environment):
        "Always updates history even if invalid action. If it is invalid revert is called"
        action = self.policy.next_action(self.get_state(), environment) # Only based on what we see, such approach doesn't generalize, it will basically understand structure of maze we gave to it
        self.action_history.append(action)
        self.pos_history.append(self.pos_history[-1] + all_actions[action])
        self.vision_history.append(self.full_vision(environment))
    
    def revert(self):
        self.pos_history.pop()
        self.pos_history.append(self.pos_history[-1])
        self.vision_history.pop()
        self.vision_history.append(self.vision_history[-1])
        self.reward_history.append(-1)


class Environment:
    def __init__(self, reward_map: np.ndarray, kill_range: int = 1) -> None:
        self.agents = []
        self.reward_map = reward_map # We generate current map online without storing since it can be restored by agent history if necessary
        self.kill_range = kill_range

        self.preys = None # Agent that tries to escape maze
        self.hunters = None # Agent that acts as an obstacle
    
    def pos_inside(self, pos: Point):
        y, x= pos.yx
        return 0 <= y < self.reward_map.shape[0] and 0 <= x < self.reward_map.shape[1]

    def random_action(self, state: State):
        actions = all_actions
        valid_actions = []
        map = state.features[:(VISIBILITY * 2 + 1) ** 2]
        # print(map)
        map = np.array(map).reshape(VISIBILITY * 2 + 1, VISIBILITY * 2 + 1)
        agent_pos = Point(map.shape[0] // 2, map.shape[1] // 2)
        for action in range(len(actions)):
            new_pos = actions[action] + agent_pos
            if map[new_pos.yx] in [EMPTY, SUBGOAL, GOAL]:
                valid_actions.append(action)
        return choice(valid_actions)

    def random_position(self):
        empty_points = []
        for y in range(self.reward_map.shape[0]):
            for x in range(self.reward_map.shape[1]):
                if self.reward_map[y, x] == EMPTY:
                    empty_points.append(Point(y, x))
        return choice(empty_points)
    
    def reset(self, agents):
        self.map = self.reward_map
        self.agents = {}
        self.agents['prey'] = []
        self.agents['hunter'] = []
        for agent in agents:
            agent.reset(self)
            self.agents[agent.role].append(agent)
    
    def update_prey(self):
        for agent in self.agents['prey']:
            if agent.has_goal:
                continue
            agent.update(self)
            
            new_pos = agent.pos_history[-1]
            old_pos = agent.pos_history[-2]
            
            if self.reward_map[new_pos.yx] == WALL:
                print(f"Agent {agent} bumped in the wall: tried to reach position {new_pos} from {old_pos}")
                agent.revert()
            elif self.reward_map[new_pos.yx] == SUBGOAL and not agent.has_subgoal:
                agent.has_subgoal = True
                agent.got_subgoal = len(agent.vision_history)
                agent.reward_history.append(5)
            elif self.reward_map[new_pos.yx] == GOAL and agent.has_subgoal:
                agent.has_goal = True
                agent.got_goal = len(agent.vision_history)
                agent.reward_history.append(10)
            else:
                agent.reward_history.append(-0.1)

        return True
    
    def update(self):
        self.update_prey()

        all_win = True
        for agent in self.agents['prey']:
            all_win = all_win and agent.has_goal
        return not all_win

    def play(self, agents, f_before=None, f_after=None, render=None, cooldown: float = 0.0, max_steps: int = None):
        self.reset(agents)
        agent = self.agents['prey'][0]
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
    
class DQLNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class ValueAction:
    def __init__(self, model):
        "To train existing model we give model as input"
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def value(self, state: State, action):
        self.model(state.features + [action])
    
    def argmax(self, state: State):
        best_action = None
        best_res = None
        for action in range(len(all_actions)):
            val = self.model(torch.tensor(state.features + [action], dtype=torch.float32))
            if best_action is None or best_res is None or val > best_res:
                best_action = action
                best_res = val
        return best_action
    
    def update(self, current_state, action, next_state, reward):
        prediction = self.value(current_state, action)
        target = reward + self.model(torch.tensor(next_state, [self.argmax(next_state)], dtype=torch.float32))
        loss = self.loss_fn(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Policy:
    "Epsilon policy, it is here just for convenience"
    def __init__(self, value_action: ValueAction, epsilon: int = 0.1):
        self.value_action = value_action
        self.epsilon = epsilon
    
    def next_action(self, state, env: Environment):
        if uniform(0, 1) < self.epsilon:
            return env.random_action(state)
        return self.value_action.argmax(state)

class DQLModel:
    def __init__(self, model, env: Environment, max_episode_step=100, action_memory=2, epsilon: int = 0.1, dim: int = 0.9, alpha: int = 1) -> None:
        self.value_action = ValueAction(model) # value action function
        self.env = env # environment
        self.max_episode_step = max_episode_step
        self.dim = dim # diminishing factor
        self.alpha = alpha # step size
        self.epsilon = epsilon # epsilon in policy
        self.action_memory = action_memory # how many actions it includes in state

        # self.total_delta = 0
        # self.count_delta = 0
    
    def b_update(self):
        self.current_state = self.agent.get_state()
    
    def a_update(self):
        action = self.agent.action_history[-1]
        reward = self.agent.reward_history[-1]
        current_state = self.current_state
        next_state = self.agent.get_state()

        self.value_action.update(current_state, action, next_state, reward)

    
    def train(self, max_episodes=10):
        for i in range(max_episodes):
            # self.total_delta = 0
            # self.count_delta = 0
            start_pos = self.env.random_position() # to support exploration
            self.agent = Agent(start_pos, Policy(self.value_action, epsilon=self.epsilon))

            self.env.play(agents=[self.agent], f_before=self.b_update, f_after=self.a_update, max_steps=self.max_episode_step)

            if i % 100 == 0:
                print(f"Subgoal: {self.agent.has_subgoal};\nGoal: {self.agent.has_goal}")
        return True

model = DQLNetwork(input_size=input_size, output_size=output_size)
env = Environment(maze_map)
qlearning = DQLModel(model=model, env=env, max_episode_step=100)
print(qlearning.train(max_episodes=200000))