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

BUFFER_SIZE = args.buffer_size#   # replay buffer size
BATCH_SIZE = args.batch_size#128        # minibatch size
GAMMA = args.gamma#0.99            # discount factor
TAU = args.tau #1e-3              # for soft update of target parameters
LR_ACTOR = args.actor_learn_rate #1e-5         # learning rate of the actor
LR_CRITIC = args.critic_learn_rate # 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = args.update_every #1        # how many steps to take before updating target networks
num_updates = args.num_updates             # update num of update
noise_factor = args.noise_factor                              # noise decay process
noise_factor_decay = args.noise_factor_decay                  # noise decay
noise_sigma=args.noise_sigma
device = args.device

print("device=",device)

class MADDPG():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, n_agnets, random_seed):
        """Initialize an Agent object.
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        n_agnets (int): numer of agents
        random_seed (int): random seed"""
        self.state_size = state_size
        self.action_size = action_size
        self.n_agnets = n_agnets
        self.seed = random.seed(random_seed)
        self.noise_factor = noise_factor

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size*self.n_agnets, action_size*self.n_agnets, random_seed).to(device)
        self.critic_target = Critic(state_size*self.n_agnets, action_size*self.n_agnets, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.hard_copy(self.actor_target, self.actor_local)
        self.hard_copy(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Step counter for Agent
        self.t_step = 0

    #def step(self, state, action, reward, next_state, done):
    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

                    
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t % UPDATE_EVERY == 0:
            experiences = self.memory.sample()
            for _ in range(num_updates):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_factor * self.noise.sample() #decay noise
            #action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        actor_target(state) -> action
        critic_target(state, action) -> Q-value
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor"""

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        actions_next=actions_next.to('cuda')
        Q_targets_next = self.critic_target(next_states, actions_next)
        rewards=rewards.to('cuda')
        dones=dones.to('cuda')
        states=states.to('cuda')
        actions=actions.to('cuda')

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
        
        # ---------------------------- decrease noise ---------- ------------- #
        self.noise_factor -= noise_factor_decay
        self.noise.reset()
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_copy(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=noise_sigma):
        """Initialize parameters and noise process."""
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        return torch.tensor(self.state * self.scale).float()
