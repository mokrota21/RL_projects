from rl import ValueAction, EPolicy
from environment import Agent, Environment, WALL_M, SUBGOAL_M, GOAL_M
from typing import List
import numpy as np
import torch.nn as nn
from metrics import MetricsVisualizer

# Maze Parameters
STATE_SIZE = 100
#hyper parameters
EPSILON = 0.6
EPSILON_DECAY = 0.9999
HIDDEN_SIZE = 128
BATCH_SIZE = 32
BUFFER_SIZE = 100
DIM = 1.0
ALPHA = 0.0001
NUM_EPISODES = 1000
MAX_STEPS = 100

def train_dql(env: Environment, value_action: ValueAction, plotter: MetricsVisualizer, num_episodes: int, max_steps: int = 100, print_every: int = 1) -> List[float]:
    # RL
    policy = EPolicy(value_action=value_action, initial_epsilon=EPSILON, decay=EPSILON_DECAY)
    episode_rewards = []
    
    for episode in range(num_episodes):
        print('-' * 100)
        agent = Agent(policy=policy)
        agent.reset(env)
        env.reset(agents=[agent])
        print(env.map)

        current_state = agent.get_state()
        episode_reward = 0

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
            episode_reward += reward
            
            # Visualisation
            plotter.update_metrics(loss=loss, reward=episode_reward, goal=agent.has_goal, subgoal=agent.has_subgoal, epsilon=policy.epsilon)

            if not updated:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    plotter.save()
    value_action.save('best_model.pth')
    return episode_rewards

value_action = ValueAction(state_size=STATE_SIZE, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, dim=DIM, alpha=ALPHA)

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

plotter = MetricsVisualizer()

train_dql(env=env, plotter=plotter, value_action=value_action, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)