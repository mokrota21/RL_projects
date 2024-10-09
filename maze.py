from design import State, Action, Policy, Environment

class MazeState(State):
    def __init__(self, y, x, subgoal, enemies) -> None:
        self.yx = (y, x)

    def __eq__(self, other) -> bool:
        if isinstance(other, MazeState):
            return self.yx == other.yx
        return False

    def __hash__(self) -> int:
        return hash(self.yx)
    
    def __str__(self) -> str:
        return str(self.yx)

    def __repr__(self) -> str:
        return str(self.yx)

class MazeAction(Action):
    def __new__(cls, name: str):
        if name.lower() == "right":
            return super().__new__(Right)
        elif name.lower() == "left":
            return super().__new__(Left)
        elif name.lower() == "up":
            return super().__new__(Up)
        elif name.lower() == "down":
            return super().__new__(Down)
        return super().__new__(cls)

    def __init__(self, name: str) -> None:
        pass

    def do_action(self, state):
        return state
    
    def __eq__(self, other) -> bool:
        return isinstance(other, type(self))
    
    def __hash__(self) -> int:
        return hash(type(self))
    
    def __str__(self) -> str:
        name = {Left: 'left', Right: 'right', Up: 'up', Down: 'down'}
        return name[type(self)]
    
    def __repr__(self) -> str:
        name = {Left: 'left', Right: 'right', Up: 'up', Down: 'down'}
        return name[type(self)]
    
    @classmethod
    def all_actions(cls):
        return [Left("left"), Right('right'), Up('up'), Down('down')]
    
class Left(MazeAction):
    def do_action(self, state):
        y, x = state.yx
        x -= 1
        return MazeState(y, x)

class Right(MazeAction):
    def do_action(self, state: MazeState):
        y, x = state.yx
        x += 1
        return MazeState(y, x)

class Up(MazeAction):
    def do_action(self, state: MazeState):
        y, x = state.yx
        y -= 1
        return MazeState(y, x)

class Down(MazeAction):
    def do_action(self, state: MazeState):
        y, x = state.yx
        y += 1
        return MazeState(y, x)