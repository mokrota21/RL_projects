from abc import ABC, abstractmethod

class Policy:
    pass

class ValueAction:
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

class Policy(ABC):
    def __init__(self, initial: dict = {}, epsilon: float = 0.01) -> None:
        self.policy = initial
        self.epsilon = epsilon

    @abstractmethod
    def next_action(self, state: State, env: Environment) -> Action:
        pass

class ValueAction(ABC):
    def __init__(self, initial: dict = {}, default: float = 0.0) -> None:
        self.value_action = initial
        self.default = default

    @abstractmethod
    def value(self, state: State, action: Action) -> Action:
        pass