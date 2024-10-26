from environment import Policy
from random import choice

class RandomPolicy(Policy):
    def __init__(self):
        pass

    def next_action(self, env):
        return choice([0, 1, 2, 3])

