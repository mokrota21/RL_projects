from environment import Environment, WALL_M, Agent
import numpy as np
from agents import AgentVision
from train import train_dql, EPolicy
from rl import ValueAction

from time import sleep
def testing(env: Environment, hyper_params_list: list, num_episodes: int = 20, max_steps: int = 100):
    count = 0
    for params in hyper_params_list:
        print(f"{count} out of {len(hyper_params_list)}")
        value_action = ValueAction(**params)
        policy = EPolicy(value_action=value_action, initial_epsilon=0.5, decay=0.9999)
        agent = AgentVision(policy=policy, visibility=10)
        metrics_path = "vision" + str(count) + '.json'
        model_path = 'vision' + str(count) + '.json'
        train_dql(env=env, agent=agent, value_action=value_action, num_episodes=200, max_steps=100,
                        save_model=model_path, save_metrics=metrics_path, random=True, print_every=10)
        # train_dql(env=env, value_action=value_action, num_episodes=num_episodes, max_steps=max_steps, print_every=1000000)
        count += 1
        sleep(2)

env = Environment()

alpha_values = [0.0001, 0.001]
tau_values = [0.0001, 0.001]
n = [1, 10]
architectures = ['exp_hidden', 'one_hidden']
steps_per_update = [10, 100, 1000]

params_table = [alpha_values, tau_values, architectures, steps_per_update, n]
params_names = "alpha tau architecture steps_per_update n".split(' ')
# state_size=STATE_SIZE, architecture=ARCHITECTURE, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, steps_per_update=STEPS_PER_UPDATE, dim=DIM, alpha=ALPHA, tau=TAU, n=10)
dim = 0.9
buffer_size = 100000
state_size = 100 * 5
batch_size = 128
default_vals = [dim, buffer_size, state_size, batch_size]
default_names = "dim buffer_size state_size batch_size".split(' ')

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

testing(env=env, hyper_params_list=params_list, num_episodes=200)

# print(len(params_list))
