import numpy as np
from random import uniform, choice
import torch
import torch.nn as nn
import random
import numpy as np
from time import sleep
from collections import deque
import pyxel
import matplotlib.pyplot as plt
from threading import Thread
from copy import deepcopy
import os

### Reward Map tiles meaning
EMPTY = 0
SUBGOAL = -1
GOAL = -2
WALL = -3
UNOBSERVED = -4
AGENT = -5
###

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

VISIBILITY = 2 # how many tiles around itself agent can see

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
maze_map = np.array(maze_map, dtype=np.int64)
input_size = maze_map.shape[0] * maze_map.shape[1] + 1 + 1 + 2 # visions + actions + has_subgoal + next_action + position
output_size = 1

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

def list_arrays_to_list(np_list):
    features = []
    for ar in np_list:
        features += numpy_to_list(ar)
    return features

class State:
    "Put all features as a list"
    def __init__(self, features: list = []):
        self.features = features

    def to_tensor(self):
        feature_list = []
        for feature in self.features:
            l = feature
            if isinstance(feature[0], np.ndarray):
                l = list_arrays_to_list(feature)
            elif not isinstance(feature, list):
                raise(f"Passed non list element to features: {feature}")
            feature_list += l
        return torch.tensor(feature_list, dtype=torch.float32).squeeze(0).to(device)

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
        self.observation_map = None
        self.action_history = None
        self.reward_history = None
        self.total_reward = None
        self.death = None
        self.has_subgoal = None
        self.has_goal = None
        self.got_subgoal = -1
        self.got_goal = -1
    
    def reset(self, environment):
        self.pos_history = deque([self.initial_pos])

        self.observation_map = np.ones(shape=environment.reward_map.shape, dtype=int) * UNOBSERVED
        self.update_vision(environment)

        self.action_history = deque([-1])

        self.reward_history = []
        self.total_reward = 0
        self.death = False
        self.has_goal = False
        self.has_subgoal = False
        self.got_subgoal = -1
        self.got_goal = -1

    def update_vision(self, environment):
        map = environment.reward_map
        pos = self.pos_history[-1].yx
        range = max(pos[0] - self.visibility, 0), min(pos[0] + self.visibility + 1, map.shape[0]), \
            max(pos[1] - self.visibility, 0), min(pos[1] + self.visibility + 1, map.shape[1])
        self.observation_map[range[0]:range[1], range[2]:range[3]] = map[range[0]:range[1], range[2]:range[3]]

    def full_vision(self, environment):
        map = environment.reward_map
        vis_shape = map.shape
        vis_shape = (vis_shape[0] + self.visibility * 2, vis_shape[1] + self.visibility * 2)
        visibility_map = np.ones(shape=vis_shape, dtype=int) * WALL
        visibility_map[self.visibility:-self.visibility, self.visibility:-self.visibility] = map
        pos = self.pos_history[-1].yx
        visibility_map = visibility_map[pos[0]:2 * self.visibility + pos[0] + 1, pos[1]: 2 * self.visibility + pos[1] + 1]
        return visibility_map
    
    def get_state(self):
        y, x = self.pos_history[-1].yx
        return State([[self.observation_map], [int(self.has_subgoal)], [y, x]]) # this way agent remembers actions it's already performed
    
    def update(self, environment):
        "Always updates history even if invalid action. If it is invalid revert is called"
        self.update_vision(environment)
        action = self.policy.next_action(self.get_state(), environment) # Only based on what we see, such approach doesn't generalize, it will basically understand structure of maze we gave to it
        self.action_history.append(action)
        self.action_history.popleft()
        self.pos_history.append(self.pos_history[-1] + all_actions[action])
    
    def revert(self):
        self.pos_history.pop()
        self.pos_history.append(self.pos_history[-1])
        self.reward_history.append(-10)


class Environment:
    def __init__(self, reward_map: np.ndarray, kill_range: int = 1) -> None:
        self.agents = []
        self.reward_map = reward_map # We generate current map online without storing since it can be restored by agent history if necessary
        self.kill_range = kill_range

        self.preys = None # Agent that tries to escape maze
        self.hunters = None # Agent that acts as an obstacle
    
    def valid_actions(self, pos: Point):
        valid_actions = []
        for i in range(len(all_actions)):
            action = all_actions[i]
            new_pos = pos + action
            if self.reward_map[new_pos.yx] != WALL:
                valid_actions.append(i)
        return valid_actions
                

    def pos_inside(self, pos: Point):
        y, x= pos.yx
        return 0 <= y < self.reward_map.shape[0] and 0 <= x < self.reward_map.shape[1]

    def random_action(self, pos: Point):
        return choice(self.valid_actions(pos))

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
    
    def update_prey(self, hit_wall):
        for agent in self.agents['prey']:
            if agent.has_goal:
                continue
            agent.update(self)
            
            new_pos = agent.pos_history[-1]
            old_pos = agent.pos_history[-2]
            
            if self.reward_map[new_pos.yx] == WALL:
                if hit_wall:
                    print(f"Agent {agent} bumped in the wall: tried to reach position {new_pos} from {old_pos}")
                agent.revert()
                agent.total_reward -= 10
            elif self.reward_map[new_pos.yx] == SUBGOAL and not agent.has_subgoal:
                agent.has_subgoal = True
                agent.reward_history.append(5)
                agent.total_reward += 5
            elif self.reward_map[new_pos.yx] == GOAL and agent.has_subgoal:
                agent.has_goal = True
                agent.reward_history.append(10)
                agent.total_reward += 10
            else:
                agent.reward_history.append(-0.1)
                agent.total_reward -= 0.1

        return True
    
    def update(self, hit_wall=False):
        self.update_prey(hit_wall=hit_wall)

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


ENV = Environment(maze_map)
    
class DQLNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(input_size, output_size),
        #     nn.Tanh(),
        #     nn.Linear(output_size, output_size)
        # ).to(device)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_size)
        ).to(device)

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(input_size, 1000),
        #     nn.Tanh(),
        #     nn.Linear(1000, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 8),
        #     nn.Tanh(),
        #     nn.Linear(8, 4),
        #     nn.Tanh(),
        #     nn.Linear(4, 2),
        #     nn.Tanh(),
        #     nn.Linear(2, output_size)
        # ).to(device)
    
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class ValueAction:
    def __init__(self, model, dim=0.9, alpha=0.01):
        "To train existing model we give model as input. It is assumed that output of model is of size of 4"
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.dim = dim

    def value(self, state: State, action):
        state.features.append([action])
        output = self.model(state.to_tensor())
        state.features.pop()
        return output

    def batch_value(self, batch):
        return self.model(batch)

    def best(self, state: State):
        self.model.eval()
        best_value = None
        best_action = None
        y, x = state.features[-1]
        pos = Point(y, x)
        valid_actions = ENV.valid_actions(pos)

        with torch.no_grad():
            for action in valid_actions:
                state.features.append([action])
                q_value = self.model(state.to_tensor())
                if best_value is None or best_action is None or q_value.item() > best_value.item():
                    best_value = q_value
                    best_action = action
                state.features.pop()
        self.model.train()
        # best_value = None
        # best_action = None
        # for action in range(len(all_actions)):
        #     state.features.append([action])
        #     q_value = self.model(state.to_tensor())
        #     if best_value is None or best_action is None or q_value.item() > best_value.item():
        #         best_value = q_value
        #         best_action = action
        #     state.features.pop()
        # self.model.train()
        return best_action, best_value

    def argmax(self, state: State):
        return self.best(state)[0]

    def max(self, state: State):
        return self.best(state)[1]

    def update(self, batch: list):
        self.model.train()
        assert len(batch[0]) == len(batch[1]) and len(batch[0]) == len(batch[2])
        
        b_state_action = []
        for i in batch[0]:
            state, action = i
            state.features.append([action])
            b_state_action.append(state.to_tensor())
        
        a_state = batch[1]
        rewards = batch[2]
        assert len(a_state) == len(rewards)
        target = []

        for i in range(len(a_state)):
            reward = rewards[i]
            next_state = a_state[i]
            target.append(reward + self.dim * self.max(next_state))

        target = torch.stack(target)
        train = torch.stack(b_state_action)
        prediction = self.batch_value(train)
        loss = self.loss_fn(prediction, target)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

class Policy:
    "Epsilon policy, it is here just for convenience"
    def __init__(self, value_action: ValueAction, initial_epsilon = 0.1, decay = 0.999, min_epsilon = 0.01):
        self.value_action = value_action
        self.epsilon = initial_epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
    
    def next_action(self, state: State, env: Environment):
        if random.random() < self.epsilon:
            y, x = state.features[-1]
            pos = Point(y, x)
            return env.random_action(pos)
        return self.value_action.argmax(state)
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
    
class DetPolicy:
    def __init__(self, value_action: ValueAction, epsilon: int = 0.1):
        self.value_action = value_action
        self.epsilon = epsilon
    
    def next_action(self, state, env: Environment):
        return self.value_action.argmax(state)

class DQLVisualization:
    def __init__(self, window_size=50):
        self.episode_rewards = []
        # self.episode_lengths = []
        self.episode_losses = []
        self.moving_avg_reward = deque(maxlen=window_size)

        # Enable interactive mode
        plt.ion()
        self.fig, (self.ax1, self.ax3) = plt.subplots(2, 1, figsize=(10, 15))
        self.fig.tight_layout()

        plt.subplots_adjust(hspace=0.5)

    def add_data(self, reward, length, loss):
        self.episode_rewards.append(reward)
        # self.episode_lengths.append(length)
        self.episode_losses.append(loss)
        self.moving_avg_reward.append(reward)

    def get_stats(self):
        avg_reward = np.mean(self.moving_avg_reward) if self.moving_avg_reward else 0
        avg_loss = np.mean(self.episode_losses[-10:]) if self.episode_losses else 0
        return avg_reward, avg_loss

    def plot_progress(self, filename=None):
        # Clear previous plots
        self.ax1.cla()
        # self.ax2.cla()
        self.ax3.cla()

        # Plot episode rewards
        self.ax1.plot(self.episode_rewards, label='Reward')
        self.ax1.set_title('Episode Rewards')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Total Reward')

        # # Plot episode lengths
        # self.ax2.plot(self.episode_lengths, label='Episode Length')
        # self.ax2.set_title('Episode Lengths')
        # self.ax2.set_xlabel('Episode')
        # self.ax2.set_ylabel('Steps')

        # Plot episode losses
        self.ax3.plot(self.episode_losses, label='Loss')
        self.ax3.set_title('Episode Losses')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Loss')

        if filename:
            self.fig.savefig(filename)

        # Redraw the updated plots
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

import keyboard

class DQLModel:
    def __init__(self, value_action: ValueAction, env: Environment, max_episode_step=100, action_memory=2, batch_size=100, 
                 initial_epsilon=0.1, decay=0.999, min_epsilon=0.01) -> None:
        self.value_action = value_action # value action function
        self.env = env # environment
        self.max_episode_step = max_episode_step
        self.initial_epsilon = initial_epsilon # epsilon in policy
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.action_memory = action_memory # how many actions it includes in state
        self.batch_size = batch_size
        self.batch = [[], [], []] # it will have 3 lists: 1 list with states+action before update, another list with states after udpate and last list with rewards

        self.visualizer = DQLVisualization()
    
    def b_update(self):
        self.current_state = self.agent.get_state()
    
    def a_update(self):
        action = deepcopy(self.agent.action_history[-1])
        reward = deepcopy(self.agent.reward_history[-1])
        current_state = self.current_state
        next_state = self.agent.get_state()
        if len(self.batch[0]) >= self.batch_size:
            self.loss = self.value_action.update(self.batch)
            self.episode_loss = self.loss.item()
            self.batch = [[], [], []]
        else:
            self.batch[0].append((current_state, action))
            self.batch[1].append(next_state)
            self.batch[2].append(reward)
        
        self.total_reward += reward

    
    def train(self, max_episodes=10):
        for i in range(max_episodes):
            # self.total_delta = 0
            # self.counter = 0
            # start_pos = Point(1, 1)
            start_pos = self.env.random_position() # to support exploration
            self.agent = Agent(start_pos, Policy(self.value_action, initial_epsilon=self.initial_epsilon, decay=self.decay, min_epsilon=self.min_epsilon))

            self.total_reward = 0
            self.episode_loss = 0

            self.env.play(agents=[self.agent], f_before=self.b_update, f_after=self.a_update, max_steps=self.max_episode_step)
            
            self.visualizer.add_data(self.total_reward, len(self.agent.action_history), self.episode_loss)

            if i % 100 == 0:
                avg_reward, avg_loss = self.visualizer.get_stats()
                print(f"Episode: {i}")
                print(f"Avg Reward: {avg_reward:.2f}")
                print(f"Avg Loss: {avg_loss:.7f}")
                print(f"Subgoal: {self.agent.has_subgoal};\nGoal: {self.agent.has_goal}")
                self.visualizer.plot_progress()
            self.agent.policy.decay_epsilon()
            
            if keyboard.is_pressed('q'):
                break

        # self.visualizer.plot_progress(show=True)

            # if i % 100 == 0:
            #     loss = self.loss.item()
            #     print(f"Episode: {i}")
            #     print(f"loss: {loss:>7f}")
            #     print(f"Subgoal: {self.agent.has_subgoal};\nGoal: {self.agent.has_goal}")
        return True
    
#####################################################################################################

if __name__ == "__main__":
    ### hyperparameters
    max_episode_step = 50
    action_memory = 2
    batch_size = 10
    initial_epsilon = 0.1
    decay = 0.999
    min_epsilon = 0.01
    dim = 0.9
    alpha = 0.01
    max_episodes = 20000
    ###
    def save_plot_with_incremented_filename(base_filename):
        # Start with the original filename
        filename = base_filename + '.png'
        counter = 1
        
        # Increment the filename if it already exists
        while os.path.exists(filename):
            filename = f"{base_filename}({counter}).png"
            counter += 1
        
        return filename

    env = Environment(maze_map)
    mode = input("Train or Play? T/P: ")
    if mode == "T":
        model_path = input("Give path to model: ") + '.pth'
        model = DQLNetwork(input_size=input_size, output_size=output_size).to(device)
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True))
        except:
            pass
        value_action = ValueAction(model, dim=dim, alpha=alpha)
        qlearning = DQLModel(value_action=value_action, env=env, max_episode_step=max_episode_step, action_memory=action_memory, batch_size=batch_size, 
                             initial_epsilon=initial_epsilon, decay=decay, min_epsilon=min_epsilon)
        print(qlearning.train(max_episodes=max_episodes))

        filename = input('Enter where to save it: ')
        torch.save(model.state_dict(), filename + '.pth')
        filename = save_plot_with_incremented_filename(filename)
        qlearning.visualizer.plot_progress(filename=filename + '.png')
        print("Saved PyTorch Model State")
    elif mode == "P":
        model_path = input("Give path to model: ") + '.pth'
        model = DQLNetwork(input_size, output_size).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        value_action = ValueAction(model, dim=dim, alpha=alpha)

    def render(environment: Environment):
        map = environment.reward_map.copy()
        for agent in environment.agents['prey']:
            map[agent.pos_history[-1].yx] = -13
        print(map)

    COL_BACKGROUND = 3
    COL_WALL = 4
    COL_AGENT = 2
    COL_SUBGOAL = 0
    COL_GOAL = 11
    COL_ERROR = 8
    PIXEL = 20

    policy = Policy(value_action)
    agent = Agent(pos=Point(1, 1), policy=policy)
    env.reset(agents=[agent])

    rewards = []

    def plot_rewards():
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots()
        while True:
            ax.clear()
            ax.plot(rewards)
            ax.set_title("Rewards Over Time")
            plt.draw()
            plt.pause(0.1)  # Update plot every 0.1 seconds

    def pyxel_render_env():
        map = env.reward_map.copy()
        for agent in env.agents['prey']:
            map[agent.pos_history[-1].yx] = -13
        
        height, width = map.shape[0], map.shape[1]  # Grid dimensions
        color_mapping = {WALL: COL_WALL, EMPTY: COL_BACKGROUND, -13: COL_AGENT, SUBGOAL: COL_SUBGOAL, GOAL: COL_GOAL, 1: COL_ERROR}

        for x_m in range(width):  # Iterate over the grid cells, not the pixel-level coordinates
            for y_m in range(height):
                obj = map[y_m, x_m]
                color = color_mapping[obj]
                # Draw a rectangle of size PIXEL x PIXEL for each grid cell
                pyxel.rect(x_m * PIXEL, y_m * PIXEL, PIXEL, PIXEL, col=color)

    def pyxel_render_perspective():
        map = agent.vision_history[-1].copy()
        map[agent.visibility, agent.visibility] = -13
        
        height, width = map.shape[0], map.shape[1]  # Grid dimensions
        color_mapping = {WALL: COL_WALL, EMPTY: COL_BACKGROUND, -13: COL_AGENT, SUBGOAL: COL_SUBGOAL, GOAL: COL_GOAL, 1: COL_ERROR}

        for x_m in range(width):  # Iterate over the grid cells, not the pixel-level coordinates
            for y_m in range(height):
                obj = map[y_m, x_m]
                color = color_mapping[obj]
                # Draw a rectangle of size PIXEL x PIXEL for each grid cell
                pyxel.rect(x_m * PIXEL, y_m * PIXEL, PIXEL, PIXEL, col=color)
    
    def pyxel_render_observation():
        map = deepcopy(agent.observation_map)
        map[agent.pos_history[-1].yx] = AGENT
        
        height, width = map.shape[0], map.shape[1]  # Grid dimensions
        color_mapping = {WALL: COL_WALL, EMPTY: COL_BACKGROUND, AGENT: COL_AGENT, SUBGOAL: COL_SUBGOAL, GOAL: COL_GOAL, UNOBSERVED: COL_ERROR}

        for x_m in range(width):  # Iterate over the grid cells, not the pixel-level coordinates
            for y_m in range(height):
                obj = map[y_m, x_m]
                color = color_mapping[obj]
                # Draw a rectangle of size PIXEL x PIXEL for each grid cell
                pyxel.rect(x_m * PIXEL, y_m * PIXEL, PIXEL, PIXEL, col=color)

    HEIGHT = 50 * env.reward_map.shape[0]
    WIDTH = 50 * env.reward_map.shape[1]

    pyxel.init(
                WIDTH, HEIGHT, title="Maze", fps=10, display_scale=1, capture_scale=60
            )

    count = 0
    def pyxel_update():
        global count, env
        env.update(hit_wall=True)
        if not agent.has_goal:
            count += 1
            rewards.append(agent.total_reward)
        else:
            env.reset([agent])
            count = 0
        print(count)

    plot_thread = Thread(target=plot_rewards)
    plot_thread.daemon = True  # Daemonize thread so it closes with the program
    # plot_thread.start()

    pyxel.run(pyxel_update, pyxel_render_observation)
