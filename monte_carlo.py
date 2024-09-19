import numpy as np
from abc import ABC, abstractmethod

class Policy:
    pass

class ValueAction:
    pass

class Environment:
    pass

class State(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

class Action(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    def do_action(self, state):
        return state

# States are used in dictionary
class MazeState(State):
    def __init__(self, y, x) -> None:
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

# Actions just do what needed, if not inside maze should be checked in environment
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

class MazeReward:
    def __init__(self) -> None:
        pass

    def reward(self, state: MazeState, action: MazeAction):
        n_state = action.do_action(state)

class Environment(ABC):
    @abstractmethod
    def all_actions(self, state: State):
        pass

    @abstractmethod
    def reward(self, state: State, action: Action):
        pass
    
    @abstractmethod
    def random_state(self):
        pass

    @abstractmethod
    def random_action(self, state: State):
        pass

    @abstractmethod
    def episode(self, policy: Policy, max_actions=100):
        pass

from random import randint, uniform, choice

class EnvironmentMaze(Environment):
    def __init__(self, reward_map) -> None:
        self.reward_map = reward_map
    
    def reward(self, state: State, action: Action):
        return self.reward_map[action.do_action(state).yx]
    
    def random_state(self):
        y_max, x_max = self.reward_map.shape
        
        y = randint(0, y_max - 1)
        x = randint(0, x_max - 1)

        return MazeState(y, x)
    
    def all_actions(self, state: State):
        actions = [Left(""), Down(""), Up(""), Right("")]
        y_max, x_max = self.reward_map.shape
        valid_actions = []

        # filter actions
        for action in actions:
            n_state = action.do_action(state)
            y, x = n_state.yx
            if 0 <= y < y_max and 0 <= x < x_max:
                valid_actions.append(action)
        
        return valid_actions
    
    def random_action(self, state: State):
        valid_actions = self.all_actions(state)
        return choice(valid_actions)

    def episode(self, policy: Policy, start_state: State = None, max_actions=100):
        if start_state is None:
            current_state = self.random_state()
        else:
            current_state = start_state
        
        state_action_reward = []
        for _ in range(max_actions):
            action = policy.next_action(current_state, self)
            reward = self.reward(current_state, action)
            state_action_reward.append((current_state, action, reward))
            current_state = action.do_action(current_state)
        return state_action_reward
    
    def render(self, state: State):
        maze = ""
        for y in range(self.reward_map.shape[0]):
            floor = ""
            for x in range(self.reward_map.shape[1]):
                floor += '0' if (y, x) != state.yx else '*'
            maze += floor
            maze += '\n'
        print(maze)
    
    def play(self, policy: Policy, start_state: State = None, max_actions=100, press_to_move=True):
        if start_state is None:
            current_state = self.random_state()
        else:
            current_state = start_state
        print(current_state)
        
        state_action_reward = []
        self.render(current_state)
        for _ in range(max_actions):
            if press_to_move:
                input("Enter")
            action = policy.next_action(current_state, self)
            reward = self.reward(current_state, action)
            current_state = action.do_action(current_state)
            self.render(current_state)
            print(reward, current_state.yx)
        return state_action_reward

# initial is dictionary (maybe empty) where keys are tuples of state and action. default should define default value if (state, action) not explored yet.
class ValueAction:
    def __init__(self, initial: dict = {}, default: float = 0.0) -> None:
        self.value_action = initial
        self.default = default
    
    def value(self, state: State, action: Action) -> float:
        return self.value_action.get((state, action), self.default)

# initial defines initial policy (maybe empty) where keys ar actions. We suppose that default if random action in environment
class Policy:
    def __init__(self, initial: dict = {}, epsilon: float = 0.01) -> None:
        self.policy = initial
        self.epsilon = epsilon

    def next_action(self, state: State, env: Environment) -> Action:
        if uniform(0, 1) < self.epsilon:
            return env.random_action(state)
        return self.policy.get(state, env.random_action(state))

def sum_l(t_sum, l, r):
    if l == 0:
        return t_sum[r]
    return t_sum[r] - t_sum[l - 1]

class MonteCarloModel:
    def __init__(self, policy: Policy, value_action: ValueAction, env: Environment, max_episode_step=100, dim=0.9) -> None:
        self.policy = policy # policy
        self.value_action = value_action # value action function
        self.env = env # environment
        self.returns = {} # list of returns for monte carlo. each value is tuple, first number is total, second amount of encounters
        self.max_episode_step = max_episode_step
        self.dim = 0.9 # diminishing factor
    
    def train(self, max_episodes=10):
        for _ in range(max_episodes):
            episode = self.env.episode(policy=self.policy, max_actions=self.max_episode_step)

            t_sum = []
            current = 0
            for _, _, reward in episode[::-1]:
                current += reward
                t_sum.append(current)
                current *= self.dim
            t_sum.reverse()
            
            visited = {}
            visited_state = {}
            for i in range(len(episode)):
                state, action, reward = episode[i]
                if visited.get((state, action), True):
                    visited[(state, action)] = False
                    
                    existed_record = self.returns.get((state, action), (0, 0))
                    total, encounter = (existed_record[0] + t_sum[i], existed_record[1] + 1)
                    self.returns[(state, action)] = (total, encounter)
                    self.value_action.value_action[(state, action)] = total / encounter

                if visited_state.get(state, True):
                    visited_state[state] = False
                    
                    valid_actions = self.env.all_actions(state)
                    best_action = valid_actions[0]
                    for action in valid_actions:
                        if self.value_action.value(state, action) > self.value_action.value(state, best_action):
                            best_action = action

                    self.policy.policy[state] = best_action
        
        return True
                
            
        

        
