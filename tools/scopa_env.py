import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tools.scopone_scientifico_sim import *
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
        
        # index of the card
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,40), dtype=np.float32)

        self.state = self.game.get_player_state(self.player)

        self.reset()
        pass

    

    def step(self, action):
        print(f"Action {action}")
        if not self.action_valid(action):
            return self.state, -1, False, {}
        state, reward, done, info = self.game.gym_step(player=self.player, action=action)
        next_state = state.flatten()  # Flastten the next state

        return next_state, reward, done, info
    
    def action_valid(self, action):
        actions = self.game.get_player_actions(self.player)
        #print(f"Invalid action for action {action} {actions[action]} and actions {actions} and {actions[action]}")
        deck = Deck().deal(40)
        
        return actions[action] == 1 and str(deck[action]) in [str(card) for card in self.player.hand]


    
    def reset(self):
        self.game.reset()
        self.game.deal_initial_hands()
        self.player = self.game.players[self.player_number]
        self.state = self.game.get_player_state(self.player)
        self.state = self.state.flatten()  # Flatten the state to a 1D array

        return self.state
    
    def render(self):
        raise NotImplementedError
    

if __name__ == "__main__":
    env = ScopaEnv()
    print(env.action_space)
    print(env.state)
    print(env.action_valid(np.argmax(env.action_space.sample())))