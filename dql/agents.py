from environment import Agent, AGENT_M, EMPTY_M, WALL_M, GOAL_M, UNOBSERVED_M
import numpy as np

class AgentVision(Agent):
    def encoder(self):
        rows, columns = self.observation_map.shape
        tile_type = [EMPTY_M, WALL_M, GOAL_M, UNOBSERVED_M, AGENT_M]
        res = np.zeros((rows, columns, len(tile_type)), dtype=np.int32)
        for i in range(len(tile_type)):
            tile = tile_type[i]
            res[self.observation_map == tile, i] = 1
        return res

