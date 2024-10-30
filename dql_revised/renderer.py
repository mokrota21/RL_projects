from environment import EMPTY_M, WALL_M, SUBGOAL_M, GOAL_M, UNOBSERVED_M, AGENT_M
import numpy as np
import pyxel
from typing import Callable

### Tile colors
EMPTY_C = 7
WALL_C = 0
SUBGOAL_C = 2
GOAL_C = 6
UNOBSERVED_C = 13
AGENT_C = 8

color_mapping = {
    EMPTY_M: EMPTY_C,
    WALL_M: WALL_C,
    SUBGOAL_M: SUBGOAL_C,
    GOAL_M: GOAL_C,
    UNOBSERVED_M: UNOBSERVED_C,
    AGENT_M: AGENT_C
}

def update():
    pass

class Renderer:
    def __init__(self, map: np.ndarray, update: Callable = update, tile_size: int = 10):
        self.update = update
        self.tile_size = tile_size
        self.map = map

        height, width = map.shape
        self.window_width = width * tile_size
        self.window_height = height * tile_size

        pyxel.init(self.window_width, self.window_height, title='Maze', fps=10, quit_key=pyxel.KEY_Q)
        pyxel.run(update=self.update, draw=self.draw)
    
    def get_tile(self, y, x):
        t_y = (self.tile_size * y, self.tile_size * (y + 1))
        t_x = (self.tile_size * x, self.tile_size * (x + 1))

        return t_y, t_x
    
    def color_tile(self, y, x, color):
        slice_y, slice_x = self.get_tile(y, x)

        for y in range(slice_y[0], slice_y[1]):
            for x in range(slice_x[0], slice_x[1]):
                pyxel.pset(x, y, color)

    def draw(self):
        height, width = self.map.shape
        for y in range(height):
            for x in range(width):
                tile_type = self.map[y, x]
                color = color_mapping[tile_type]

                self.color_tile(y, x, color)
