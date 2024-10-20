from environment import Environment, WALL_M
import numpy as np
from train import train_dql
from rl import ValueAction

from time import sleep
def testing(env: Environment, hyper_params_list: list, num_episodes: int = 20, max_steps: int = 100):
    count = 0
    for params in hyper_params_list:
        print(f"{count} out of {len(hyper_params_list)}")
        value_action = ValueAction(**params)
        metrics_path = "metrics_data" + str(count) + '.json'
        train_dql(env=env, value_action=value_action, num_episodes=num_episodes, max_steps=max_steps, print_every=1000000)
        count += 1
        sleep(2)


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

alpha_values = [0.0001, 0.001, 0.01]
tau_values = [0.001, 0.01, 0.05]
batch_size_values = [32, 128, 512]

params_table = [batch_size_values, alpha_values, tau_values]
params_names = "batch_size alpha tau".split(' ')

dim = 0.9
buffer_size = 10000
state_size = 100 * 6
default_vals = [dim, buffer_size, state_size]
default_names = "dim buffer_size state_size".split(' ')

params_list = []
for i in range(len(alpha_values) ** len(params_table)):
    combination = [0] * len(params_table)
    count = 0
    while i > 0:
        combination[count] = i % len(alpha_values)
        i = i // 5
        count += 1

    combo = {}
    for i, val in enumerate(combination):
        param_name = params_names[i]
        param_values = params_table[i]
        combo[param_name] = param_values[val]

    for i in range(len(default_names)):
        name = default_names[i]
        val = default_vals[i]
        combo[name] = val
    
    params_list.append(combo)

# testing(env=env, hyper_params_list=params_list, num_episodes=5)

print(params_list[10])
