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
        

class Maze:
    """The game environment"""

    def __init__(self, players, timedelta_per_action) -> None:
        """Initiate pyxel, set up initial game variables and run."""
        pyxel.init(
            WIDTH, HEIGHT, title="Snake!", fps=120, display_scale=20, capture_scale=6
        )
        self.players = players
        self.timedelta_per_action = timedelta_per_action
        self.reset()
        pyxel.run(self.update, self.draw)
    
    def reset(self):
        """Initiate key variables (direction, snake, apple, score, etc.)"""
        
        all_pos = list(range(WIDTH * HEIGHT))
        positions = sample(all_pos, len(self.players))
        positions = list(map(lambda pos: Point(pos % WIDTH, pos // WIDTH), positions))
        self.positions = {}
        for i in range(len(positions)):
            player = self.players[i]
            position = positions[i]
            self.positions[player] = position
        
        self.color = list(range(4, 16))
        self.death = [False] * len(self.players)
        self.death_count = 0
        self.roles = [Role('prey') for i in range(len(self.players) - 1)]
        self.roles.append('hunter')
        self.time = datetime.now()
    
    def update(self):
        """Update logic of game.
        Updates the snake and checks for scoring/win condition."""

        if len(self.death) - self.death_count > 1:
            t_delta = (datetime.now() - self.time)
            # print(t_delta)
            if t_delta < self.timedelta_per_action:
                for i in range(len(self.players)):
                    player = self.players[i]
                    pos = self.positions[player]
                    player.update(pos)
                # map(lambda player: player.update(), self.players)
            else:
                for i in range(len(self.players)):
                    self.update_player(i)
                self.time = datetime.now()

        if pyxel.btn(pyxel.KEY_Q):
            pyxel.quit()

        if pyxel.btnp(pyxel.KEY_R) or pyxel.btnp(pyxel.GAMEPAD1_BUTTON_A):
            self.reset()

    def point_inside(self, point: Point):
        y, x = point.y, point.x
        return 0 <= y < HEIGHT and 0 <= x < WIDTH
        

    def update_player(self, i):
        """Move the snake based on the direction."""

        player = self.players[i]
        old_pos = self.positions[player]
        direction = player.direction
        new_pos = Point(old_pos.x + direction.x, old_pos.y + direction.y)

        if self.point_inside(new_pos):
            self.positions[player] = new_pos
        player.direction = Point(0, 0)
    
    def draw(self):
        """Draw the background, snake, score, and apple OR the end screen."""

        if len(self.death) - self.death_count > 1:
            pyxel.cls(col=COL_BACKGROUND)
            self.draw_players()

    def draw_players(self):
        """Draw the snake with a distinct head by iterating through deque."""
        for i in range(len(self.positions)):
            player = self.players[i]
            pos = self.positions[player]
            color = self.color[i]
            pyxel.pset(pos.x, pos.y, col=color)

class DummyPolicy(design.Policy):
    def next_action(self, state: maze.State, env: design.Environment) -> maze.Action:
        return maze.MazeAction(choice(['up', 'down', 'right', 'left']))

timedelta_between_action = timedelta(seconds=1) / 10
print(timedelta_between_action)
Maze([RealPlayer(), BotPlayer(DummyPolicy())], timedelta_between_action)