from rl import ValueAction, EPolicy
from environment import Agent, Environment, WALL_M, SUBGOAL_M, GOAL_M
from typing import List
import numpy as np
import torch.nn as nn
from metrics import MetricsVisualizer
import keyboard

#hyper parameters
EPSILON = 0.6
EPSILON_DECAY = 0.9999

def train_dql(env: Environment, value_action: ValueAction, num_episodes: int, max_steps: int = 100, print_every: int = 1, 
              save_metrics: str = None, save_model: str = None, plot=False) -> List[float]:
    # RL
    policy = EPolicy(value_action=value_action, initial_epsilon=EPSILON, decay=EPSILON_DECAY)
    reward_total = 0
    reward_count = 0
    
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
    
    if plot:
        plotter.close()
        if save_metrics:
            plotter.save(save_metrics)
        else:
            plotter.save()
    if save_model:
        value_action.save(save_model)
    return reward_total
