from rl import ValueAction, EPolicy
from environment import Agent, Environment, WALL_M, SUBGOAL_M, GOAL_M
from typing import List
import numpy as np
import torch.nn as nn
from metrics import MetricsVisualizer
import keyboard
import os
import json

#hyper parameters

def get_unique_filename(filename):
    if not os.path.exists(filename):
            return filename
            
    name, ext = os.path.splitext(filename)
    counter = 1
        
    while True:
        new_path = f"{name}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def train_dql(env: Environment, value_action: ValueAction, num_episodes: int, max_steps: int = 100, print_every: int = 1, epsilon=0.6, epsilon_decay=0.999,
              save_metrics: str = None, save_model: str = None, plot=False, save_per=100) -> List[float]:
    # RL
    policy = EPolicy(value_action=value_action, initial_epsilon=epsilon, decay=epsilon_decay)
    reward_total = 0
    reward_count = 0
    data = {'loss': [],
            'reward': [],
            'goal': [],
            'subgoal': [],
            'epsilon': []
            }
    data_count = 0
    
    if plot:
        plotter = MetricsVisualizer()
    stop = False
    
    for episode in range(num_episodes):
        # print('-' * 100)
        agent = Agent(policy=policy)
        env.reset(agents=[agent])
        # print(env.map)

        current_state = agent.get_state()
        episode_reward_total = 0

        for _ in range(max_steps):
            # RL
            updated = env.update()
            action = agent.action_history[-1]
            next_state = agent.get_state()
            reward = agent.reward_history[-1]
            value_action.memory.push(state=current_state, action=action, reward=reward, next_state=next_state)

            loss = value_action.learn()
            policy.decay_epsilon()

            current_state = next_state
            reward_total += reward
            reward_count += 1
            episode_reward_total += reward

            # Caching
            data_count += 1
            if data_count % save_per == 0:
                data['loss'].append(loss)
                data['reward'].append(reward_total / reward_count)
                data['goal'].append(agent.has_goal)
                data['subgoal'].append(agent.has_subgoal)
                data['epsilon'].append(policy.epsilon)
                data_count = 0
            
            # Visualisation
            if plot:
                plotter.update_metrics(loss=loss, reward=reward_total / reward_count, goal=agent.has_goal, subgoal=agent.has_subgoal, epsilon=policy.epsilon)

            if keyboard.is_pressed('q'):
                stop = True
            if not updated:
                break
        
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}, Episode's Reward: {episode_reward_total:.2f}, Subgoal: {agent.has_subgoal}, Goal: {agent.has_goal}, Exploration: {policy.epsilon}")
        if stop:
            break
    
    if save_metrics is not None:
        unique_path = get_unique_filename(save_metrics)
        with open(unique_path, 'w') as file:
            json.dump(data, file, indent=4)
    if plot:
        plotter.close()
    if save_model:
        value_action.save(save_model)
    return reward_total
