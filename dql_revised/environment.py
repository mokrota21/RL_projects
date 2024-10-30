import numpy as np
from random import sample, choice, uniform, shuffle, randint
from time import sleep

### Map representation
EMPTY_M = 0
WALL_M = 1
SUBGOAL_M = 2
GOAL_M = 3
UNOBSERVED_M = 4
AGENT_M = 5
### Rewards
EMPTY_R = -0.1
WALL_R = -1
SUBGOAL_R = 50
GOAL_R = 50
###

class State:
    pass
class Maze:
    pass
class Agent:
    pass

class Point:
    def __init__(self, y, x):
        self.yx = (y, x)
    
    def __add__(self, other):
        if isinstance(other, Point):
            y, x = self.yx
            o_y, o_x = other.yx
            return Point(y + o_y, x + o_x)
        raise TypeError('Trying to add non point object')
    
    def __sub__(self, other):
        if isinstance(other, Point):
            y, x = self.yx
            o_y, o_x = other.yx
            return Point(y - o_y, x - o_x)
        raise TypeError('Trying to substract non point object')
    
    def __str__(self):
        return str(self.yx)
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash(self.yx)
    
    def __eq__(self, other):
        return isinstance(other, Point) and self.yx == other.yx

class Maze:
    def __init__(self, map: np.ndarray, goal_pos: Point = None, subgoal_pos: Point = None, start_pos: Point = None):
        self.map = map
        self.goal_pos = goal_pos
        self.subgoal_pos = subgoal_pos
        self.start_pos = start_pos
    
    def empty_pos(self):
        empty_coords = np.where(self.map == EMPTY_M)
        empty_points = [Point(empty_coords[0][i], empty_coords[1][i]) for i in range(len(empty_coords[0]))]
        return empty_points
    
    def random_pos(self, k=1):
        empty_points = self.empty_pos()
        return sample(empty_points, k)

    def random_maze(self):
        height, width = self.map.shape
        new_map = np.full((height, width), WALL_M)

        def inside(pos):
            y, x = pos.yx
            return  1 <= y < height - 1 and 1 <= x < width - 1

        def surrounded(f, t):
            assert inside(f) and inside(t)
            res = True
            directions = [Point(-1, 0), Point(1, 0), Point(0, 1), Point(0, -1)]
            for direction in directions:
                n = t + direction
                if inside(n) and n != f:
                    res = res and new_map[n.yx] == WALL_M
            return res

        def neighbours(pos):
            assert inside(pos)
            y, x = pos.yx
            print(new_map)

            options = [Point(y - 1, x), Point(y + 1, x), Point(y, x - 1), Point(y, x + 1)]
            res = []
            for option in options:
                if inside(option) and new_map[option.yx] == WALL_M and surrounded(pos, option):
                    res.append(option)
            return res

        y_range = list(range(1, height - 1))
        x_range = list(range(1, width - 1))
        random_start = Point(choice(y_range), choice(x_range))
        new_map[random_start.yx] = EMPTY_M

        order = [random_start]
        current = None

        while len(order) > 0:
            current = order[-1]
            new_candidates = neighbours(current)
            if new_candidates:
                candidate = choice(new_candidates)
                new_map[candidate.yx] = EMPTY_M
                order.append(candidate)

            else:
                order.pop(-1)
                
            
        return new_map

    def reset(self, random=False):
        if random:
            self.goal_pos, self.subgoal_pos = self.random_pos(k=2)
            self.map = self.random_maze()

def main():
    empty_map = np.full((10, 10), WALL_M, dtype=np.int64)
    maze = Maze(empty_map)
    from renderer import Renderer
    renderer = Renderer(maze.random_maze())

if __name__ == '__main__':
    main()