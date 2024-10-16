from base import State, device, torch
from environment import Agent, Environment, Point, Policy, all_actions, deque, np
import torch.nn as nn
from random import random
import matplotlib as plt
from copy import deepcopy

class DQLNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(input_size, output_size),
        #     nn.Tanh(),
        #     nn.Linear(output_size, output_size)
        # ).to(device)
        self.stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_size)
        ).to(device)
    
    def forward(self, x):
        return self.stack(x)

class ValueAction:
    def __init__(self, model, loss_fn=nn.MSELoss(), dim=0.9, alpha=0.01):
        "To train existing model we give model as input. It is assumed that output of model is of size of 4"
        self.model: nn.Module = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.dim = dim
    
    def method_mode(self, mode, method, *args, **kwargs):
        training = self.model.training
        if mode == 'eval':
            self.model.eval()
        elif mode == 'train':
            self.model.train()
        else:
            raise(f"Passed invalid mode: {mode}")
        
        output = method(*args, **kwargs)

        if training:
            self.model.train()
        else:
            self.model.eval()
        
        return output

    def value(self, state: State, action, train=False):
        state.features.append([action])
        output = self.model(state.to_tensor())
        state.features.pop()
        return output

    def batch_value(self, batch):
        return self.model(batch)

    def best(self, state: State):
        best_value = None
        best_action = None

        for action in all_actions:
            state.features.append([action])
            q_value = self.model(state.to_tensor())
            if best_value is None or best_action is None or q_value.item() > best_value.item():
                best_value = q_value
                best_action = action
            state.features.pop()
        
        return best_action, best_value

    def argmax(self, state: State):
        return self.best(state)[0]

    def max(self, state: State):
        return self.best(state)[1]

    def update(self, batch: list):
        "Every entry of training consists of 3 elements: state before action (combined with action), state after action, reward"
        self.model.train()
        assert len(batch[0]) == len(batch[1]) and len(batch[0]) == len(batch[2])
        
        before_states = batch[0]
        after_states = batch[1]
        rewards = batch[2]
        assert len(after_states) == len(rewards)
        target = []

        for i in range(len(after_states)):
            reward = rewards[i]
            next_state = after_states[i]
            approx_reward = reward + self.dim * self.method_mode('eval', self.max, state=next_state)
            target.append(approx_reward)

        target = torch.stack(target)
        train = torch.stack(before_states)
        prediction = self.method_mode('train', self.batch_value, batch=train)
        loss = self.loss_fn(prediction, target)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

class EPolicy(Policy):
    "Epsilon policy, communicates agent with Value Action function"
    def __init__(self, value_action: ValueAction, initial_epsilon = 0.1, decay = 0.999, min_epsilon = 0.01):
        self.value_action = value_action
        self.epsilon = initial_epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
    
    def next_action(self, state: State, env: Environment):
        if random() < self.epsilon:
            return env.random_action()
        return self.value_action.method_mode("eval", self.value_action.argmax, state=state)
    
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