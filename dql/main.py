from train import train_dql, ValueAction, Environment, EPolicy
from environment import WALL_M, Point, Agent
from agents import AgentVision
import numpy as np

# Maze Parameters
STATE_SIZE = 500 # observation size * types of tiles
#hyper parameters
BATCH_SIZE = 128
BUFFER_SIZE = 100000
DIM = 0.9
ALPHA = 0.0001
NUM_EPISODES = 10000
MAX_STEPS = 100
TAU = 1.0
# First 250
# EPSILON = 0.6
# EPSILON_DECAY = 0.9999
# After 250
EPSILON = 0.1
EPSILON_DECAY = 0.999999
DROPOUT = 0.2
STEPS_PER_UPDATE = 128
ARCHITECTURE = 'exp_hidden'

value_action = ValueAction(state_size=STATE_SIZE, architecture=ARCHITECTURE, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, steps_per_update=STEPS_PER_UPDATE, dim=DIM, alpha=ALPHA, tau=TAU, n=1)
try:
    value_action.load('full_vision_long_run_exp.pth')
    print('loaded model')
except:
    print('failed to load')

# maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
#             [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
#             [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#             [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
#             [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
#             [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
#             [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
#             [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
# replace_1 = lambda x: WALL_M if x == 1 else x
# maze_map = list(map(lambda x: list(map(replace_1, x)), maze_map))
# maze_map = np.array(maze_map, dtype=np.int64)
# env = Environment(map=maze_map, subgoal_pos=Point(5, 3), goal_pos=Point(8, 7))
env = Environment()
policy = EPolicy(value_action=value_action, initial_epsilon=EPSILON, decay=EPSILON_DECAY)
agent = AgentVision(policy=policy)
train_dql(env=env, agent=agent, value_action=value_action, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS,
          save_model='full_vision_long_run_exp.pth', save_metrics="full_vision_long_run_exp.json", plot=False, random=True)