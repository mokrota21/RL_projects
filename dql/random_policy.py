from environment import Agent, Environment, WALL_M, SUBGOAL_M, GOAL_M, Policy
from metrics import MetricsVisualizer
from typing import List
from random import choice
import numpy as np
import json

class RandomPolicy(Policy):
    def __init__(self):
        pass

    def next_action(self, env):
        return choice([0, 1, 2, 3])

def random_play(env: Environment, num_episodes: int, max_steps: int = 100, print_every: int = 1,
              save_metrics: str = None, save_model: str = None) -> List[float]:
    # RL
    policy = RandomPolicy()
    reward_total = 0
    reward_count = 0
    
    plotter = MetricsVisualizer()
    stop = False
    avg_rewards = {'avg_rewards': []}
    
    for episode in range(num_episodes):
        # print('-' * 100)
        agent = Agent(policy=policy)
        env.reset(agents=[agent], random=False)
        # print(env.map)
        episode_reward_total = 0

        for _ in range(max_steps):
            # RL
            updated = env.update()
            action = agent.action_history[-1]
            
            
            reward = agent.reward_history[-1]
            reward_total += reward
            reward_count += 1
            avg_rewards['avg_rewards'].append(reward_total / reward_count)
            episode_reward_total += reward
        
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}, Episode's Reward: {episode_reward_total:.2f}")
        if stop:
            break
    
    plotter.close()
    with open('random_rewards.json', 'w') as json_file:
        json.dump(avg_rewards, json_file, indent=4) 
    return reward_total

maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
replace_1 = lambda x: WALL_M if x == 1 else x
maze_map = list(map(lambda x: list(map(replace_1, x)), maze_map))
maze_map = np.array(maze_map, dtype=np.int64)
env = Environment(map=maze_map)

random_play(env, 5, save_metrics='random.json')