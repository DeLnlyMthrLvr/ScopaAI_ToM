import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import flatten_v0, pad_observations_v0, pad_action_space_v0
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.spaces import flatten_space

# Step 1: Define MAPPO-compatible Environment Wrapper
class PettingZooSB3Env:
    def __init__(self, env):
        self.env = env
        self.agents = env.possible_agents
        self.num_envs = 1  # Required by SB3
        self.current_agent = 0
        self.reset()

    def reset(self):
        obs = self.env.reset()
        self.dones = {agent: False for agent in self.agents.index()}
        self.agent_obs = {agent: obs[agent] for agent in self.agents.index()}
        return np.array([self.agent_obs[agent] for agent in self.agents.index()])

    def step(self, actions):
        rewards = {}
        infos = {}
        for i, agent in enumerate(self.agents.index()):
            if not self.dones[agent]:
                obs, reward, done, info = self.env.step({agent: actions[i]})
                self.dones[agent] = done
                rewards[agent] = reward
                infos[agent] = info
                if done:
                    obs[agent] = self.env.reset()
                self.agent_obs[agent] = obs[agent]

        obs = np.array([self.agent_obs[agent] for agent in self.agents.index()])
        rewards = np.array([rewards[agent] for agent in self.agents.index()])
        dones = np.array([self.dones[agent] for agent in self.agents.index()])
        return obs, rewards, dones, infos

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return flatten_space(self.env.observation_space(self.agents.index()[0]))

    @property
    def action_space(self):
        return self.env.action_space(self.agents.index()[0])

# Step 2: Define Centralized Critic
class CentralizedCritic(nn.Module):
    def __init__(self, observation_space, action_space, num_agents):
        super(CentralizedCritic, self).__init__()
        self.num_agents = num_agents
        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0] * num_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observations):
        batch_size = observations.size(0)
        centralized_input = observations.view(batch_size, -1)  # Flatten agent observations
        value = self.fc(centralized_input)
        return value

# Step 3: Extend ActorCriticPolicy for MAPPO
class MAPPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, num_agents, **kwargs):
        super(MAPPOPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        self.centralized_critic = CentralizedCritic(observation_space, action_space, num_agents)

    def evaluate_actions(self, obs, actions):
        value = self.centralized_critic(obs)
        distribution = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)
        return value, log_prob, distribution.entropy()

# Step 4: Integrate Environment and MAPPO Policy
if __name__ == "__main__":
    from maenv.ma_scopa_env import MaScopaEnv  # Import the PettingZoo environment
    env = MaScopaEnv()  
    env.reset()

    # Prepare the environment
    raw_env = env
    parallel_env = aec_to_parallel(raw_env)
    wrapped_env = flatten_v0(pad_observations_v0(pad_action_space_v0(parallel_env)))

    vec_env = DummyVecEnv([lambda: PettingZooSB3Env(wrapped_env)])

    # Define the MAPPO model
    model = PPO(
        policy=MAPPOPolicy,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs={"num_agents": len(raw_env.possible_agents)},
        verbose=1
    )

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("mappo_scopone_model")

    # Test the model
    obs = vec_env.reset()
    for _ in range(100):
        actions, _ = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(actions)
        if all(dones):
            break
