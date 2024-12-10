import torch as T
from maddpg.networks import ActorNetwork, CriticNetwork
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# originally from https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients/blob/a3c294aa6834f348a7401306dff3e67919c861f5/agent.py

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_indx, checkpoint_dir, writer: SummaryWriter=None,
                alpha=0.01, beta=0.01, fc1=64, 
                fc2=64, gamma=0.95, tau=0.01, v=0):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f'agent_{agent_indx}'

        self.writer = writer

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=checkpoint_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=checkpoint_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=checkpoint_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=checkpoint_dir,
                                            name=self.agent_name+'_target_critic')

    #def get_valid_actions(self, observation):


    def choose_action(self, observation):
        state = T.tensor([np.array(observation)], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        return actions.detach().cpu().numpy()

    

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()