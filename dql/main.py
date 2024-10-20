from train import train_dql, ValueAction, Environment, WALL_M, np

# Maze Parameters
STATE_SIZE = 5 ** 2 * 5 # observation size * types of tiles
#hyper parameters
BATCH_SIZE = 32
BUFFER_SIZE = 10000
DIM = 0.9
ALPHA = 0.001
NUM_EPISODES = 10000
MAX_STEPS = 100
TAU = 0.1
DROPOUT = 0.2

# def train_dql(env: Environment, value_action: ValueAction, num_episodes: int, max_steps: int = 100, print_every: int = 1,
#               save_metrics: str = None, save_model: str = None) -> List[float]:
#     # RL
#     policy = EPolicy(value_action=value_action, initial_epsilon=EPSILON, decay=EPSILON_DECAY)
#     reward_total = 0
#     reward_count = 0
    
#     plotter = MetricsVisualizer()
    
#     for episode in range(num_episodes):
#         # print('-' * 100)
#         agent = Agent(policy=policy)
#         env.reset(agents=[agent], random=False)
#         # print(env.map)

#         current_state = agent.get_state()
#         episode_reward_total = 0

#         for _ in range(max_steps):
#             # RL
#             updated = env.update()
#             action = agent.action_history[-1]
#             next_state = agent.get_state()
#             reward = agent.reward_history[-1]
#             value_action.memory.push(state=current_state, action=action, reward=reward, next_state=next_state)

#             loss = value_action.learn()
#             policy.decay_epsilon()

#             current_state = next_state
#             reward_total += reward
#             reward_count += 1
#             episode_reward_total += reward
            
#             # Visualisation
#             plotter.update_metrics(loss=loss, reward=reward_total / reward_count, goal=agent.has_goal, subgoal=agent.has_subgoal, epsilon=policy.epsilon)

#             if not updated or keyboard.is_pressed('q'):
#                 break
        
#         if (episode + 1) % print_every == 0:
#             print(f"Episode {episode + 1}, Episode's Reward: {episode_reward_total:.2f}")
    
#     plotter.close()
#     if save_metrics:
#         plotter.save(save_metrics)
#     else:
#         plotter.save()
#     if save_model:
#         value_action.save(save_model)
#     return reward_total

value_action = ValueAction(state_size=STATE_SIZE, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, dim=DIM, alpha=ALPHA, tau=TAU)
try:
    value_action.load('simple_model.pth')
    [print('loaded model')]
except:
    print('failed to load')
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
train_dql(env=env, value_action=value_action, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, save_model='simple_model.pth')