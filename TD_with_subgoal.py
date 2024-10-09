from TD import TDModel, TDPolicy, TDValueAction, Environment
from design import State, Action, Policy
from random import sample
import numpy as np
import typing

VISIBILITY = 2
GOAL = -3
SUBGOAL = -2
WALL = -1
EMPTY = 0
# -3 - goal tile
# -2 - subgoal tile
# -1 - wall tile
# 0 - empty tile
# n - number of agents in tile

REWARD_T = 10
REWARD_B = 11
assert REWARD_B - REWARD_T >= 1

class Point:
    def __init__(self, y, x) -> None:
        self.yx = y, x
    
    def __add__(self, o):
        if isinstance(o, Point):
            y1, x1 = self.yx
            y2, x2 = o.yx
            y, x = y1 + y2, x1 + x2
            return Point(y, x)

class MazeState(State):
    pass
        

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

    def do_action(self, pos: Point):
        return pos
    
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
    
class Left(MazeAction):
    def do_action(self, pos: Point):
        add = Point(0, -1)
        return pos + add
        

class Right(MazeAction):
    def do_action(self, pos: Point):
        add = Point(0, 1)
        return pos + add

class Up(MazeAction):
    def do_action(self, pos: Point):
        add = Point(-1, 0)
        return pos + add

class Down(MazeAction):
    def do_action(self, pos: Point):
        add = Point(1, 0)
        return pos + add

class EnvironmentMaze:
    pass

class MazeAgent:
    """
    This is what is acting as an agent based on the policy
    """
    def __init__(self, policy: Policy, role: str = "prey", pos: Point = Point(0, 0)) -> None:
        self.policy = policy
        self.dead = False
        self.role = role
        self.pos = pos
        self.has_subgoal = False
        self.has_goal = False
        
        self.history = [] # state action reward state action reward state ...

    def reset(self, pos):
        self.dead = False
        self.has_goal = False
        self.pos = pos
        self.has_subgoal = False

    def put_everything(self, env: EnvironmentMaze):
        obs_state = np.zeros((VISIBILITY * 2 + 1, VISIBILITY * 2 + 1))
        for dy in range(-VISIBILITY, VISIBILITY + 1):
            for dx in range(-VISIBILITY, VISIBILITY + 1):
                pos = self.pos
                vec = Point(dy, dx)
                new_pos = pos + vec
                y_ag, x_ag = dy + VISIBILITY, dx + VISIBILITY
                observed = EMPTY
                if env.inside(new_pos):
                    observed = env.maze_state[new_pos.yx]
                else:
                    observed = WALL
                
                obs_state[y_ag, x_ag] = observed
        return obs_state

    def walls(self, obs_state):
        pass

    def update(self, env: EnvironmentMaze):
        obs_state = self.put_everything(env, obs_state)
        obs_state = self.walls(obs_state=obs_state)

        self.action = self.policy.next_action(obs_state, env)

        self.history += [obs_state, self.action]

class EnvironmentMaze(Environment):
    def inside(self, pos: Point):
        y, x = pos.yx
        y_max, x_max = self.maze_map.shape
        return 0 <= y < y_max and 0 <= x < x_max

    def __init__(self, maze_map, reward_map, agents: list[MazeAgent], kill_range: int = 2) -> None:
        self.maze_map = maze_map
        self.kill_range = 2
        self.agent_positions = {}
        for agent in agents:
            self.agent_positions[agent] = agent.pos
        self.reset(reward_map=reward_map, agents=agents)
    
    def reset(self, reward_map, agents: list[MazeAgent]):
        """
        Resets maze and agents to their initial positions
        """
        self.maze_state = self.maze_map
        self.reward_map = reward_map
        self.agents = agents
        self.prey_count = 0
        self.hunter_count = 0
        for agent in agents:
            if agent.role == 'prey':
                self.prey_count += 1
            else:
                self.hunter_count += 1
            agent.pos = self.agent_positions[agent]
            y, x = agent.pos.yx
            assert self.maze_map[y, x] != -1
            self.maze_state[y, x] += 1
        # w, h = self.maze_map.shape
        # positions = list(range(w * h))
        # positions = sample(positions, len(agents))
        # for i in range(len(self.agents)):
        #     agent = self.agents[i]
        #     y, x = positions[i] // w, positions[i] % w
        #     agent.reset(y, x)
        #     self.maze_state[y, x] = 5
    
    def reward_agent(self, agent: MazeAgent):
        reward = -1
        state = agent.put_everything(self)
        if agent.has_subgoal:
            goal = np.where(self.state == GOAL)
        else:
            goal = np.where(self.state == SUBGOAL)

        if len(goal[0]) != 0:
            gy, gx = int(goal[0]), int(goal[1])
            ay, ax = agent.pos.yx
            dy, dx = abs(gy - ay), abs(gx - ax)
            if dy - dx
            reward += (REWARD_T - dy - dx) / REWARD_B

        agent.history.append(reward)
        return reward
            
    
    def move_agents(self):
        for agent in self.agents:
            if not agent.dead and not agent.win:
                old_pos = agent.pos
                action = agent.action
                new_pos = action.do_action(old_pos)
                if self.inside(new_pos):
                    if self.maze_state[old_pos.yx] not in [SUBGOAL, GOAL]:
                        self.maze_state[old_pos.yx] -= 1
                    if self.maze_state[new_pos.yx] not in [SUBGOAL, GOAL]:
                        self.maze_state[new_pos.yx] += 1
                    else:
                        self.maze
                    agent.pos = new_pos
                
                self.reward_agent(agent, old_pos)
    
    def terminate(self):
        if self.prey_count == 0 or self.hunter_count == 0

    def death_agents(self):
        pass
                
    
    def update(self):
        self.move_agents()
        self.death_agents()

