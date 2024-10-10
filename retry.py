import numpy as np
from time import sleep

### Reward Map tiles meaning
EMPTY = 0
SUBGOAL = -1
GOAL = -2
WALL = -3
# AGENTS = n where n is number of agents on the tile
ACTION_MEMORY = 0
VISIBILITY = 10

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

def numpy_to_list(np_array):
    res = []
    for row in np_array.tolist():
        res += row
    return res

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
    def __init__(self, features) -> None:
        self.features = features
    
    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            return self.features == other.features
        return False
    
    def __hash__(self) -> int:
        return hash(tuple(self.features))
    
    def __str__(self) -> str:
        return str(self.features)

    def __repr__(self) -> str:
        return str(self.features)

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

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self))
    
    def __hash__(self) -> int:
        return hash(type(self))
    
    def __str__(self) -> str:
        name = {Left: 'left', Right: 'right', Up: 'up', Down: 'down'}
        return name[type(self)]
    
    def __repr__(self) -> str:
        name = {Left: 'left', Right: 'right', Up: 'up', Down: 'down'}
        return name[type(self)]

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
    
def all_actions():
    return [Action('up'), Action('down'), Action('left'), Action('right')]

class Agent:
    "Class that provides communication between Actions and Environment"
    def __init__(self, pos: Point, policy: Policy, value_action: ValueAction = None, role: str = 'prey', visibility: int = VISIBILITY) -> None:
        self.initial_pos = pos
        self.role = role
        self.policy = policy
        self.visibility = visibility
        self.value_action = value_action
        
        self.pos_history = None
        self.vision_history = None
        self.action_history = None
        self.reward_history = None
        self.death = None
        self.has_subgoal = None
        self.has_goal = None
    
    def reset(self, environment):
        self.pos_history = [self.initial_pos]
        self.vision_history = [self.full_vision(environment=environment)]
        self.action_history = []
        self.reward_history = []
        self.death = False
        self.has_goal = False
        self.has_subgoal = False

    def full_vision(self, environment):
        map = environment.map
        vis_shape = map.shape
        vis_shape = (vis_shape[0] + self.visibility * 2, vis_shape[1] + self.visibility * 2)
        visibility_map = np.ones(shape=vis_shape, dtype=int) * WALL
        visibility_map[self.visibility:-self.visibility, self.visibility:-self.visibility] = map
        pos = self.pos_history[-1].yx
        visibility_map = visibility_map[pos[0]:2 * self.visibility + pos[0] + 1, pos[1]: 2 * self.visibility + pos[1] + 1]
        return visibility_map
    
    def update(self, environment):
        action: Action = self.policy.next_action(State(numpy_to_list(self.vision_history[-1]) + self.action_history[-ACTION_MEMORY:]), self.value_action, environment) # Only based on what we see, such approach doesn't generalize, it will basically understand structure of maze we gave to it
        # action: Action = self.policy.next_action(State(self.vision_history[-1], self.action_history), environment) # More generalizable approach but need to finetune amount of actions we remember
        self.action_history.append(action)
        self.pos_history.append(action.do_action(self.pos_history[-1]))
        self.vision_history.append(self.full_vision(environment))
    
    def revert(self):
        self.pos_history.pop()
        self.vision_history.pop()

from random import choice

class Environment:
    def __init__(self, reward_map: np.ndarray, kill_range: int = 1) -> None:
        self.agents = []
        self.reward_map = reward_map
        self.kill_range = kill_range

        self.map = None
        self.preys = None
        self.hunters = None
    
    def pos_inside(self, pos: Point):
        y, x= pos.yx
        return 0 <= y < self.reward_map.shape[0] and 0 <= x < self.reward_map.shape[1]

    def random_action(self, state: State):
        actions = all_actions()
        valid_actions = []
        map = state.features[:(VISIBILITY * 2 + 1) ** 2]
        map = np.array(map).reshape(VISIBILITY * 2 + 1, VISIBILITY * 2 + 1)
        agent_pos = Point(map.shape[0] // 2, map.shape[1] // 2)
        for action in actions:
            new_pos = action.do_action(agent_pos)
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
            # try:
                # agent.update(self)
            # except:
            #     print(f"Failed to update agent {agent} with a role prey")
            #     return False
            
            new_pos = agent.pos_history[-1]
            old_pos = agent.pos_history[-2]
            if not self.pos_inside(new_pos):
                print(f"Invalid action from agent {agent}: tried to reach position {new_pos} from {old_pos}")
                agent.revert()
            
            if self.reward_map[new_pos.yx] == WALL:
                print(f"Agent {agent} bumped in the wall: tried to reach position {new_pos} from {old_pos}")
                agent.revert()
            elif self.reward_map[new_pos.yx] == SUBGOAL and not agent.has_subgoal:
                agent.has_subgoal = True
                agent.reward_history.append(5)
            elif self.reward_map[new_pos.yx] == GOAL and agent.has_subgoal:
                agent.has_goal = True
                agent.reward_history.append(100)
            else:
                agent.reward_history.append(-1)

        return True
    
    ### If we do 2 agents thing

    # def update_prey(self):
    #     for agent in self.agents['hunter']:
    #         try:
    #             agent.update()
    #         except:
    #             print(f"Failed to update agent {agent} with a role hunter")
    #             return False
            
    #         new_pos = agent.pos_history[-1]
    #         old_pos = agent.pos_history[-2]
    #         if not self.pos_inside(new_pos):
    #             print(f"Invalid action from agent {agent}: tried to reach position {new_pos} from {old_pos}")
    #             agent.revert()
            
    #         if self.reward_map[new_pos] == WALL:
    #             print(f"Agent {agent} bumped in the wall: tried to reach position {new_pos} from {old_pos}")
    #             agent.revert()
    #         elif self.reward_map[new_pos] == SUBGOAL:
    #             agent.has_subgoal = True
    #         elif self.reward_map[new_pos] == GOAL and agent.has_subgoal:
    #             agent.has_goal = True    

    #     return True
    
    # def death_update(self)
    
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

class ValueAction:
    "Value action is implemented with double layered dictionary. First layer consists of states and second of actions."
    def __init__(self, initial: dict = {}, default: float = 0) -> None:
        self.state_d = initial
        self.default = default
    
    def value(self, state: State, action: Action):
        action_d: dict = self.state_d.get(state, {})
        return action_d.get(action, self.default)
    
    def argmax(self, state: State) -> Action:
        best_action = None
        action_d: dict = self.state_d.get(state, {})
        if action_d:
            for action in action_d.items():
                if best_action is None or action_d[best_action] < action_d[action[0]]:
                    best_action = action[0]
        return best_action
    
    def max_value(self, state: State):
        res = None
        best_action = self.argmax(state)
        if best_action:
            action_d: dict = self.state_d[state]
            return action_d[best_action]
        return self.default

from random import uniform

class Policy:
    "Epsilon greedy policy"
    def __init__(self, initial: dict = {}, epsilon: float = 0.01) -> None:
        self.policy = initial
        self.epsilon = epsilon
    
    def next_action(self, state: State, value_action: ValueAction, env: Environment) -> Action:
        if value_action:
            if uniform(0, 1) < self.epsilon:
                return env.random_action(state)
            action = value_action.argmax(state)
            if action is None:
                action = env.random_action(state)
            return action
        return env.random_action(state)


class QLearningModel:
    def __init__(self, policy: Policy, value_action: ValueAction, env: Environment, max_episode_step=100, action_memory=2, dim=0.9, alpha=1) -> None:
        self.policy = policy # policy
        self.value_action = value_action # value action function
        self.env = env # environment
        self.max_episode_step = max_episode_step
        self.dim = dim # diminishing factor
        self.alpha = alpha # step size
        self.action_memory = action_memory # how many actions it includes in state

        self.avg_delta = 0
        self.count_delta = 0
    
    def approx(self):
        action = self.agent.action_history[-1]
        reward = self.agent.reward_history[-1]
        current_state = State(numpy_to_list(self.agent.vision_history[-2]) + self.agent.action_history[-self.action_memory - 1:-1])
        next_state = State(numpy_to_list(self.agent.vision_history[-1]) + self.agent.action_history[-self.action_memory:])
        
        self.value_action.state_d[current_state] = self.value_action.state_d.get(current_state, {})
        delta = reward + self.dim * self.value_action.max_value(next_state) - self.value_action.value(current_state, action)
        self.avg_delta = self.avg_delta * self.count_delta + delta
        self.count_delta += 1
        self.avg_delta = self.avg_delta / self.count_delta
        self.value_action.state_d[current_state][action] = self.value_action.state_d.get(current_state, {}).get(action, 0) + (
                    self.alpha * delta)

    
    def train(self, max_episodes=10):
        for i in range(max_episodes):
            current_pos = self.env.random_position()
            self.agent = Agent(current_pos, self.policy, self.value_action)

            self.env.play(agents=[self.agent], f_after=self.approx, max_steps=self.max_episode_step)

            if i % 100 == 0:
                print(f"Subgoal: {self.agent.has_subgoal};\nGoal: {self.agent.has_goal}; Avg Delta: {self.avg_delta}")
            
            self.avg_delta = 0
            self.count_delta = 0
        
        return True

# Start training from 0

policy = Policy()
value_action = ValueAction()
env = Environment(maze_map)
qlearning = QLearningModel(policy=policy, value_action=value_action, env=env, max_episode_step=1000)
print(qlearning.train(max_episodes=2000))

# Save trained model
import pickle
file_path = input("Path to save model (wihtout extension): ") + '.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(qlearning, file)

# Load trained model
# import pickle
# with open('qlearning_model.pkl', 'rb') as file:
#     qlearning = pickle.load(file)
# policy = qlearning.policy
# value_action = qlearning.value_action


def render(environment: Environment):
    map = environment.reward_map.copy()
    for agent in environment.agents['prey']:
        map[agent.pos_history[-1].yx] = -13
    print(map)

import pyxel
COL_BACKGROUND = 3
COL_WALL = 4
COL_AGENT = 2
COL_SUBGOAL = 0
COL_GOAL = 11
WIDTH = 200
HEIGHT = 200
PIXEL = 20

env = Environment(maze_map)
def pyxel_render():
    map = env.reward_map.copy()
    for agent in env.agents['prey']:
        map[agent.pos_history[-1].yx] = -13
    
    height, width = map.shape[0], map.shape[1]  # Grid dimensions
    color_mapping = {WALL: COL_WALL, EMPTY: COL_BACKGROUND, -13: COL_AGENT, SUBGOAL: COL_SUBGOAL, GOAL: COL_GOAL}

    for x_m in range(width):  # Iterate over the grid cells, not the pixel-level coordinates
        for y_m in range(height):
            obj = map[y_m, x_m]
            color = color_mapping[obj]
            # Draw a rectangle of size PIXEL x PIXEL for each grid cell
            pyxel.rect(x_m * PIXEL, y_m * PIXEL, PIXEL, PIXEL, col=color)

    

random_policy = Policy()
agent = Agent(pos=Point(1, 1), policy=random_policy)
env.reset(agents=[agent])
pyxel.init(
            WIDTH, HEIGHT, title="Maze", fps=10, display_scale=1, capture_scale=60
        )
pyxel.run(env.update, pyxel_render)
# env.play(agents=[agent], render=render, cooldown=0.1)


