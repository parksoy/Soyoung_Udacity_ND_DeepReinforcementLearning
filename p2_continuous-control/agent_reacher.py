import numpy as np
import random
import copy
from collections import namedtuple, deque
from nnmodels_reacher import Actor, Critic
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from config_settings import Args

# %load_ext autoreload
# %autoreload 2

args=Args()

BUFFER_SIZE = args.buffer_size             # replay buffer size
BATCH_SIZE = args.batch_size               # minibatch size
GAMMA = args.gamma                         # discount factor
TAU = args.tau                             # for soft update of target parameters
LR_ACTOR = args.actor_learn_rate #1e-4     # learning rate of the actor
LR_CRITIC = args.critic_learn_rate #1e-3   # learning rate of the critic
WEIGHT_DECAY = 0                           # L2 weight decay
UPDATE_EVERY = args.update_every  #        # timesteps between updates
num_updates = args.num_updates             # update num of update
noise_factor = args.noise_factor                             # noise decay process
noise_factor_decay = args.noise_factor_decay                  # noise decay
noise_sigma=args.noise_sigma

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        random_seed (int): random seed """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.noise_factor = noise_factor

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device) #33,4
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device) #33,4
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, random_seed) # Noise process
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed) # Replay memory

        self.hard_copy(self.actor_target, self.actor_local)
        self.hard_copy(self.critic_target, self.critic_local)


    def step(self, states, actions, rewards, next_states, dones, t): #, done
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

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
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise_factor * self.noise.sample() #decay noise
            #actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        actor_target(state) -> action
        critic_target(state, action) -> Q-value
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        next_states=next_states.to('cuda')

        actions_next = self.actor_target(next_states) # Get predicted next-state actions and Q values from target models
        actions_next=actions_next.to('cuda')
        Q_targets_next = self.critic_target(next_states, actions_next)
        rewards=rewards.to('cuda')
        dones=dones.to('cuda')
        states=states.to('cuda')
        actions=actions.to('cuda')
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # Compute Q targets for current states (y_i)
        Q_expected = self.critic_local(states, actions) # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad() # Minimize the loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) #grad clipping
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean() # Compute actor loss
        self.actor_optimizer.zero_grad() # Minimize the loss
        actor_loss.backward()
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
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=noise_sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
