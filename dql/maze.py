from rl import ValueAction, EPolicy
from environment import Agent, Environment, WALL_M, SUBGOAL_M, GOAL_M
from typing import List
import numpy as np
import torch.nn as nn
from metrics import MetricsVisualizer
import keyboard
import os
import json
from random_policy import RandomPolicy

def random_baseline(env: Environment, agent: Agent, num_episodes: int, max_steps: int = 100, print_every: int = 1,
              save_metrics: str = None, save_model: str = None, plot=False, save_per=100, random=True) -> List[float]:
    # RL
    reward_total = 0
    reward_count = 0
    data = {'loss': [],
            'reward': [],
            'goal': [],
            'subgoal': [],
            'epsilon': []
            }
    data_count = 0
    
    stop = False
    
    for episode in range(num_episodes):
        # print('-' * 100)
        env.reset(agents=[agent], random=random)
        # print(env.map)

        current_state = agent.get_state()
        episode_reward_total = 0
        n_reward = 0
        l_rewards = []

        for step_count in range(max_steps):
            # RL
            updated = env.update()
            # action = agent.action_history[-1]
            # next_state = agent.get_state()
            reward = agent.reward_history[-1]

            # if len(l_rewards) == n:
                # value_action.memory.push(state=current_state, action=action, reward=n_reward, next_state=next_state, terminate=not updated)

                # loss = value_action.learn()
                # agent.policy.decay_epsilon()
                
                # Caching
            data_count += 1
            if data_count % save_per == 0:
                # data['loss'].append(loss)
                data['reward'].append(episode_reward_total / (step_count + 1))
                data['goal'].append(agent.has_goal)
                data['subgoal'].append(agent.has_subgoal)
                data_count = 0
                
                # Visualisation
                # if plot:
                    # plotter.update_metrics(loss=loss, reward=reward_total / reward_count, goal=agent.has_goal, subgoal=agent.has_subgoal, epsilon=agent.policy.epsilon)

            # current_state = next_state
            reward_total += reward
            reward_count += 1
            episode_reward_total += reward

           

            if keyboard.is_pressed('q'):
                stop = True
            if not updated:
                break
        
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}, Episode's Reward: {episode_reward_total:.2f}, Subgoal: {agent.has_subgoal}, Goal: {agent.has_goal}")
        if stop:
            break
    
    with open(save_metrics, 'w') as file:
            json.dump(data, file, indent=4)
    return reward_total

r_policy = RandomPolicy()
agent = Agent(r_policy)
random_baseline(Environment(), agent, 9000, save_metrics='random_baseline.json')