import numpy as np
import random
import copy
from collections import namedtuple, deque
from nnmodels_tennis import Actor, Critic
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
from config_settings import Args

args=Args()

BUFFER_SIZE = args.buffer_size#          # replay buffer size
BATCH_SIZE = args.batch_size             # minibatch size
GAMMA = args.gamma                       # discount factor
TAU = args.tau                           # for soft update of target parameters
LR_ACTOR = args.actor_learn_rate         # learning rate of the actor
LR_CRITIC = args.critic_learn_rate       # learning rate of the critic
WEIGHT_DECAY = 0                         # L2 weight decay
UPDATE_EVERY = args.update_every         # how many steps to take before updating target networks
NUM_UPDATES = args.num_updates           # update num of update
noise_factor_decay = args.noise_factor_decay        # noise decay
NOISE_START = 1.0                        # epsilon decay for the noise process added to the actions
NOISE_DECAY = args.noise_factor_decay    # decay for for subrtaction of noise
NOISE_SIGMA = args.noise_sigma           # sigma for Ornstein-Uhlenbeck noise

device = args.device

print("device=",device)


class DDPG():
    """DDPG agent with one actor and one critic."""
    def __init__(self, id, model, action_size=2, seed=0,
                 tau=TAU,
                 lr_actor=LR_ACTOR,
                 lr_critic=LR_CRITIC,
                 weight_decay=0.0000):

        """model: model object
        action_size (int): dimension of each action
        seed (int): Random seed
        tau (float): for soft update of target parameters
        lr_actor (float): learning rate for actor
        lr_critic (float): learning rate for critic
        weight_decay (float): L2 weight decay"""

        random.seed(seed)
        self.id = id
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.critic_loss = 0
        self.actor_loss = 0
        self.noise_val = 0

        # Actor Network
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        self.hard_copy(self.actor_target, self.actor_local)
        self.hard_copy(self.critic_target, self.critic_local)

    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.reshape(1, -1)).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
        all_next_actions (list): each agent's next_action (as calculated by it's actor)
        all_actions (list): each agent's action (as calculated by it's actor)"""

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)

        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)

        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))

        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        self.critic_loss = critic_loss.item()  # for tensorboard logging

        # minimize loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()

        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_loss = actor_loss.item()  # calculate policy gradient

        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter"""

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def hard_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class MADDPG():
    """Meta agent that contains the two DDPG agents and shared replay buffer."""

    def __init__(self, action_size=2, seed=0, load_file=None,
                 n_agents=2,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 gamma=GAMMA,
                 update_every=UPDATE_EVERY,
                 noise_start=NOISE_START,
                 noise_decay=NOISE_DECAY,
                 evaluation_only=False):

        """action_size (int): dimension of each action
        seed (int): Random seed
        load_file (str): path of checkpoint file to load
        n_agents (int): number of distinct agents
        buffer_size (int): replay buffer size
        batch_size (int): minibatch size
        gamma (float): discount factor
        update_every (int): how often to update the network
        noise_start (float): initial noise weighting factor
        noise_decay (float): noise decay rate
        evaluation_only (bool): set to True to disable updating gradients and adding noise"""

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.n_agents = n_agents
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.evaluation_only = evaluation_only

        # create two agents, each with their own actor and critic
        models = [MultiAgents(n_agents=n_agents) for _ in range(n_agents)]
        self.agents = [DDPG(0, models[0]), DDPG(1, models[1])]

        # create shared replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and self.evaluation_only == False:
            if len(self.memory) > self.batch_size:
                for _ in range(NUM_UPDATES):
                    experiences = [self.memory.sample() for _ in range(self.n_agents)]
                    self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            if self.evaluation_only:
                action = agent.act(state, noise_weight=self.noise_weight, add_noise=False)
            else:
                action = agent.act(state, noise_weight=self.noise_weight, add_noise=True)
            self.noise_weight -= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector

    def learn(self, experiences, gamma):
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)


        all_actions = [] # each agent uses it's own actor to calculate actions
        for i, agent in enumerate(self.agents):
            states, _, _, _, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)

        for i, agent in enumerate(self.agents):# each agent learns from it's experience sample
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)


class MultiAgents():

    """Container for actor and critic along with respective target networks.
    Each actor takes a state input for a single agent.
    Each critic takes a concatentation of the states and actions from all agents."""

    def __init__(self, n_agents=2, state_size=24, action_size=2, seed=0):

        """n_agents (int): number of distinct agents
        state_size (int): number of state dimensions for a single agent
        action_size (int): number of action dimensions for a single agent
        seed (int): random seed"""

        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        critic_input_size = state_size*n_agents
        self.critic_local = Critic(critic_input_size, action_size*n_agents, seed).to(device)
        self.critic_target = Critic(critic_input_size, action_size*n_agents, seed).to(device)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=NOISE_SIGMA):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state
