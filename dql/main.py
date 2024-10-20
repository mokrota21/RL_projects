from train import train_dql, ValueAction, Environment

# Maze Parameters
STATE_SIZE = 10 ** 2 * 5 # observation size * types of tiles
#hyper parameters
BATCH_SIZE = 128
BUFFER_SIZE = 10000
DIM = 0.9
ALPHA = 0.0001
NUM_EPISODES = 1000
MAX_STEPS = 100
TAU = 0.001
EPSILON = 0.9
EPSILON_DECAY = 0.9999
DROPOUT = 0.2
STEPS_PER_UPDATE = 32

value_action = ValueAction(state_size=STATE_SIZE, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, steps_per_update=STEPS_PER_UPDATE, dim=DIM, alpha=ALPHA, tau=TAU)
try:
    value_action.load('steps_per_update.pth')
    [print('loaded model')]
except:
    print('failed to load')
env = Environment()
train_dql(env=env, value_action=value_action, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, 
          save_model='steps_per_update.pth', save_metrics="metrics_1.json", plot=False)