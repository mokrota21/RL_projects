import numpy as np
import random

# y, x coordinate of deleted node
def graph_connected(g, y, x):
    

def take_random_spot(checked):
    y_order = random.shuffle(list(range(len(checked))))

    for i in range(len(y_order)):
        y = y_order[i]
        x_order = []
        for x in range(len(checked[y])):
            if not checked[y, x]: 
                x_order.append(x)
        if len(x_order) > 0:
            return y, x_order[0]
    
    return False

def generate_map(x_max, y_max, max_walls):
    maze_map = np.zeros(y_max, x_max)
    maze_map[0, :] = 1
    maze_map[-1, :] = 1
    maze_map[:, 0] = 1
    maze_map[:, -1] = 1

    count_walls = 0
    while count_walls < max_walls:
        checked = [[False] * (x_max - 2)] * (y_max - 2)
        
