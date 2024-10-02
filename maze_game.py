from collections import deque, namedtuple
import maze
import design
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from random import choice, sample
import pyxel

Point = namedtuple("Point", ["x", "y"])  # Convenience class for coordinates

class Role:
    def __init__(self, name) -> None:
        assert name in ["hunter", "prey"]
        self.name = name
        

COL_BACKGROUND = 3
COL_BODY = 11
COL_HEAD = 7
COL_DEATH = 8
COL_APPLE = 8

TEXT_DEATH = ["GAME OVER", "(Q)UIT", "(R)ESTART"]
COL_TEXT_DEATH = 0
HEIGHT_DEATH = 5

WIDTH = 40
HEIGHT = 50

UP = Point(0, -1)
DOWN = Point(0, 1)
RIGHT = Point(1, 0)
LEFT = Point(-1, 0)
STAY = Point(0, 0)

class Player(ABC):
    """Basic Player class, it decides what actions to push in environment"""

    @abstractmethod
    def update(self, pos: Point):
        pass

class RealPlayer(Player):
    def __init__(self) -> None:
        self.direction = STAY
    
    def update(self, pos):
        # print('here')
        if pyxel.btn(pyxel.KEY_UP):
            self.direction = UP
        elif pyxel.btn(pyxel.KEY_DOWN):
            self.direction = DOWN
        elif pyxel.btn(pyxel.KEY_LEFT):
            self.direction = LEFT
        elif pyxel.btn(pyxel.KEY_RIGHT):
            self.direction = RIGHT
    
class BotPlayer(Player):
    def __init__(self, policy: design.Policy) -> None:
        self.policy = policy
        self.direction = STAY

    def point_to_state(self, point: Point):
        return maze.MazeState(point.y, point.x)

    def action_to_point(self, action: maze.MazeAction):
        start = maze.MazeState(0, 0)
        end = action.do_action(start)
        y_s, x_s = start.yx
        y_e, x_e= end.yx
        return Point(x_e - x_s, y_e - y_s)

    def update(self, pos: Point):
        s_state = self.point_to_state(pos)
        self.direction = self.action_to_point(self.policy.next_action(s_state, None))

class MazeGame:
    def __init__(self, players, roles, kill_distance, dt_action: timedelta) -> None:
        pyxel.init(
            WIDTH, HEIGHT, title="Maze", fps=120, display_scale=20, capture_scale=6
        )
        self.kill_distance = kill_distance
        self.players = players
        self.dt_action = dt_action
        self.reset(roles)
        pyxel.run(self.update, self.draw)
    
    def reset(self, roles):
        self.death = {}
        
        self.roles = {}

        self.positions = {}
        self.colors = {}
        color = 4
        all_positions = list(range(WIDTH * HEIGHT))
        positions = sample(all_positions, len(self.players))
        for i in range(len(self.players)):
            player = self.players[i]
            y, x = positions[i] // WIDTH, positions[i] % WIDTH
            self.positions[player] = Point(x, y)

            self.colors[player] = color
            color += 1

            self.death[player] = False

            self.roles[player] = roles[i]
        
        self.time = datetime.now()
    
    def update(self):
        t_delta = datetime.now() - self.time
        if t_delta < self.dt_action:
            for player in self.players:
                player.update(self.positions[player])
        else:
            for player in self.players:
                if not self.death[player]:
                    self.move_player(player)
                    if self.roles[player].name == "hunter":
                        self.check_kill(player)
            
            self.time = datetime.now()

        if pyxel.btn(pyxel.KEY_R):
            self.reset()
        if pyxel.btn(pyxel.KEY_Q):
            pyxel.quit()
    
    def check_kill(self, player):
        h_pos = self.positions[player]
        for o_player in self.players:
            if not self.death[o_player]:
                o_pos = self.positions[o_player]
                dist_x = abs(o_pos.x - h_pos.x)
                dist_y = abs(o_pos.y - h_pos.y)
                if self.roles[o_player].name == "prey" and dist_x + dist_y <= self.kill_distance:
                    self.death[o_player] = True



    def inside(self, point):
        return 0 <= point.x < WIDTH and 0 <= point.y < HEIGHT

    def move_player(self, player):
        direction = player.direction
        pos = self.positions[player]
        new_pos = Point(pos.x + direction.x, pos.y + direction.y)
        if self.inside(new_pos):
            self.positions[player] = new_pos
        player.direction = STAY

    
    def draw(self):
        pyxel.cls(col=COL_BACKGROUND)
        for player in self.players:
            if not self.death[player]:
                self.draw_player(player)
    
    def draw_player(self, player):
        color = self.colors[player]
        pos = self.positions[player]
        # print(pos, color, self.roles[player].name)
        pyxel.pset(pos.x, pos.y, col=color)

class DummyPolicy(design.Policy):
    def next_action(self, state: maze.State, env: design.Environment) -> maze.Action:
        return maze.MazeAction(choice(['up', 'down', 'right', 'left']))

MazeGame(players=[RealPlayer(), BotPlayer(DummyPolicy()), BotPlayer(DummyPolicy())], roles=[Role('prey'), Role('hunter'), Role('hunter')], kill_distance=2, dt_action=timedelta(seconds=1) / 10)