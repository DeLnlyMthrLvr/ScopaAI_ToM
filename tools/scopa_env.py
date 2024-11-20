import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scopone_scientifico_sim import *
from tqdm import tqdm

class ScopaEnv(gym.Env):
    def __init__(self, player_number: int = 0):
        if player_number < 0 or player_number > 3:
            raise ValueError("Player number must be between 0 and 3")
        self.player_number = player_number
        self.game = ScoponeGame()
        self.game.deal_initial_hands()
        self.player = self.game.players[self.player_number]

        # 1 player 40 possible cards + 40 possible captures + 40 possible cards on table
        self.observation_space = spaces.MultiBinary((3,40))
        
        # 50 possible actions: 40 cards
        self.action_space = spaces.MultiBinary(40)

        self.state = self.game.get_player_state(self.player)

        self.reset()
        pass

    

    def step(self, action):
        state, reward, done, info = self.game.gym_step(player=self.player, action=action)

        return state, reward, done, info


    
    def reset(self):
        self.game.reset()
        self.game.deal_initial_hands()
        self.player = self.game.players[self.player_number]
        self.state = self.game.get_player_state(self.player)

        return self.state
    
    def render(self):
        raise NotImplementedError
    

if __name__ == "__main__":
    env = ScopaEnv()
    print(env.action_space)
    print(env.state)