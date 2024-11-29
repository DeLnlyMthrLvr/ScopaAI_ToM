import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tools.scopone_scientifico_sim import *
from tqdm import tqdm

class ScopaEnv(gym.Env):
    def __init__(self, player_number: int = 0, playing_players: int = 1):
        if player_number < 0 or player_number > 3:
            raise ValueError("Player number must be between 0 and 3")
        self.player_number = player_number
        self.game = ScoponeGame()
        self.game.deal_initial_hands()
        self.player = self.game.players[self.player_number]
        self.playing_players = playing_players

        # 1 player 40 possible cards + 40 possible captures + 40 possible cards on table
        self.observation_space = spaces.MultiBinary((3,40))
        
        # index of the card
        self.action_space = spaces.Box(0, 1, shape=(1,40))

        self.state = self.game.get_player_state(self.player)

        self.reset()
        pass

    

    def step(self, action, player, v= 0):
        
        if not self.action_valid(action, player, v=v):
            raise ValueError(f"Invalid action {action} for player {player}")
        state, reward, done, info = self.game.gym_step(player=player, action=action, v=v)
        for i in range(4 - self.playing_players):
            self.game.random_step(self.game.players[i+1], action=None, v=v) # +1 because we are 0

        done = False
        if [len(self.game.players[i].hand) == 0 for i in range(4)] == [True, True, True, True]:
            winning_threshold = 21
            bonus = 0
            self.game.last_capture.capture(self.game.table, _with=None)
            self.game.table = []
            eval = self.game.evaluate_round(self.game.players, v=v)
            self.game.match_points[0] += eval[0]
            self.game.match_points[1] += eval[1]
            side1_score, side2_score = self.game.match_points
            if abs(side1_score-side2_score) == 1 and min(side1_score, side2_score) >= winning_threshold-1:
                if v >= 1: print(f'[MATCH] DEUCE! old threshold: {winning_threshold} new threshold: {max(side1_score, side2_score) + 1}')
                winning_threshold = max(side1_score, side2_score) + 1

            if side1_score >= winning_threshold and side2_score < side1_score - 1:
                if v >= 0: print(f'[MATCH] Side 1 wins with {side1_score} points!')
                done = True
                bonus = 5
            elif side2_score >= winning_threshold and side1_score < side2_score - 1:
                if v >= 0: print(f'[MATCH] Side 2 wins with {side2_score} points!')
                done = True
                bonus = -5

            if v == -7 and done: print(f'[RL] Game is over! {self.game.match_points[0]}|{self.game.match_points[1]}')
            if not done:
                if v == -7: print(f'[RL] Resetting game! {self.game.match_points[0]}|{self.game.match_points[1]}')
                return self.reset(soft=True), reward + bonus, done, {}


        if done and self.game.match_points[0]>self.game.match_points[0]:
            return None, 5, done, {}
        elif done and not self.game.match_points[0]>self.game.match_points[0]:
            return None, -5, done, {}
        
        
        
        return self.game.get_player_state(player), reward, done, info
    
    def action_valid(self, action, player, v= 0):
        if action is None or player is None:
            return False
        actions = self.game.get_player_actions(player)
        if v ==-1: print(f"[ACTION VALIDATOR] Action for action {action} and actions {actions} and hand {player.hand}")
        return actions[action]>=1

    def get_player(self, intex):
        if intex < 0 or intex > 3:
            raise ValueError("Player number must be between 0 and 3. Recieved: ", intex)
        return self.players[intex]


    
    def reset(self, soft=False):
        self.game.reset(soft=soft)
        self.game.deal_initial_hands()
        self.player = self.game.players[self.player_number]
        self.state = self.game.get_player_state(self.game.players[0])
        self.state = self.state.flatten()  # Flatten the state to a 1D array

        return self.state
    
    def render(self):
        raise NotImplementedError
    

if __name__ == "__main__":
    env = ScopaEnv()
    print(env.action_space)
    print(env.state)