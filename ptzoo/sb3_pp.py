import numpy as np
from maenv.ma_scopa_env import MaScopaEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.conversions import aec_to_parallel
from tqdm import tqdm
import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tlogger import TLogger

import os
import glob

SIDE = 1

class SB3ActionMaskWrapper(BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset()

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)

    def action_masks(self):
        """Separate function used in order to access the action mask."""
        return self.get_action_mask()

def sanity_check(mask):
    # Checks that the mask is not malformed. Functions only for a novel enviroment (all with starting cards)
    for m in mask:
        assert sum(m) == 10

    for m in range(len(mask[0])):
        mask_sum = np.sum([mask[i][m] for i in range(len(mask))])
        assert mask_sum == 1 

def mask_fn(env):
    return env.get_action_mask()


def train_action_mask(env_fn, writer_log, steps=10_000, seed=42, **env_kwargs):
    """Train a single model to play as each agent in a Scopone Scientifico game environment using invalid action masking."""
    env = env_fn

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env.unwrapped)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log=writer_log)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_ToM1_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()

def eval_action_mask(env_fn, num_games=10000, render_mode=None, side= SIDE):
    # Evaluate a trained agent vs a random agent
    env = env_fn
    
    if side == 0:
        sidet = ['player_0', 'player_2']
        nsidet = ['player_1', 'player_3']
    else:
        sidet = ['player_1', 'player_3']
        nsidet = ['player_0', 'player_2']

    print(
        f"Starting evaluation vs a random agent.\n\t!Random! agent will play as side: {side} with players: {sidet}\n\t!Trained! agent will be players: {nsidet}"
    )

    try:
        policies = glob.glob(f"{env.metadata['name']}*.zip")
        latest_policy = max(
            policies, key=os.path.getctime
        )
        tomZero = policies[1]
        print(f"Loading policy: {latest_policy} amd {tomZero}")
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(tomZero)
    model_TOM = MaskablePPO.load(latest_policy)

    

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in tqdm(range(num_games), desc="Playing games"):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            observation, action_mask = obs, info['action_mask']

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    if winner == 'player_0' or winner == 'player_2':
                        scores['player_2'] += env.rewards[winner] + env.rewards['player_0']
                        scores['player_0'] += env.rewards[winner] + env.rewards['player_2']
                    elif winner == 'player_1' or winner == 'player_3':
                        scores['player_3'] += env.rewards[winner] + env.rewards['player_1']
                        scores['player_1'] += env.rewards[winner] + env.rewards['player_3']

                      # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                
                if agent not in sidet:
                    #act = env.action_space(agent).sample(action_mask.astype(np.int8))
                    act = int(model_TOM.predict(
                            observation, action_masks=action_mask
                        )[0]
                    )
                else:
                    # Note: PettingZoo expects integer actions # TODO: readapt!!!! and check the results of what is going on
                    act = int(model.predict(
                            observation[:3], action_masks=action_mask
                        )[0]
                    )

            env.step(act)
            tlogger.add_tick()
    scoresp = env.roundScores()
    env.close()



    plt.show()


    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[0]] / sum(scores.values())
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return total_rewards, winrate, scores


if __name__ == '__main__':

    

    experiment_name = f"[0VS1]testing_ToM_s{SIDE}_10k_mappo_scopa_{time.strftime('%m%d-%H%M%S')}"

    #experiment_name = f"Training_ToM_3M_mappo_scopa_{time.strftime('%m%d-%H%M%S')}"

    tlogger = TLogger(f"runs/{experiment_name}")

    env = MaScopaEnv(tlogger=tlogger, render_mode='human')
    #env = aec_to_parallel(env)
    env.reset()

    #train_action_mask(env_fn=env, writer_log=tlogger.get_log_dir(), steps=3_000_000, seed=42)

    eval_action_mask(env, num_games=10_000)

    plt.bar([f'player_{i}' for i in range(4)], tlogger.scopas_log)

    plt.show()
    

