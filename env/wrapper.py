import numpy as np
import pygame
from env.game_env import RacingEnv

class CarRacingWrapper:
    def __init__(self, seed=0):
        pygame.init()
        np.random.seed(seed)
        self.env = RacingEnv()
        self.n_actions = 5

    def _draw(self, action):
        self.env.render(action) 
        pygame.display.flip()  
        pygame.event.pump()   

    def reset(self):
        self.env.reset()   
        state_vec, _, _ = self.env.step(0) 
        self._draw(0) 
        return np.array(state_vec, dtype=np.float32)

    def step(self, action):
        state_vec, r, done = self.env.step(action)
        self._draw(action) 
        return np.array(state_vec, dtype=np.float32), r, done, {}

    def close(self):
        self.env.close()
        pygame.quit()