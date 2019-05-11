import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_settings import Args

# %load_ext autoreload
# %autoreload 2

args=Args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device=",device)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=args.layer_sizes[0], fc2_units=args.layer_sizes[1]): #States[0] (33,)
        """Initialize parameters and build model.
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.bn0 = nn.BatchNorm1d(state_size).to(device)
        self.fc1 = nn.Linear(state_size, fc1_units).to(device) #33
        self.bn1 = nn.BatchNorm1d(fc1_units).to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units).to(device)
        self.bn2 = nn.BatchNorm1d(fc2_units).to(device)
        self.fc3 = nn.Linear(fc2_units, action_size).to(device) #4
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if str(type(state))=="<class \'numpy.ndarray\'>":
            states = torch.from_numpy(states).float().to(device)

        x = self.bn0(state).to(device)
        x = F.relu(self.bn1(self.fc1(x))) #x = F.relu(self.fc1(state))
        x = F.relu(self.bn2(self.fc2(x))) #x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, seed, fcs1_units=args.layer_sizes[0], fc2_units=args.layer_sizes[1]):
        """Initialize parameters and build model.
        Params
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size).to(device)
        self.fcs1 = nn.Linear(state_size, fcs1_units).to(device)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units).to(device)
        self.fc3 = nn.Linear(fc2_units, 1).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        if str(type(state))=="<class \'numpy.ndarray\'>":
            state = torch.from_numpy(state).float().to(device)
        state = self.bn0(state)
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)




class DoubleAgent():

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

        # Print model
        print("self.actor_local=", self.actor_local)
        print("self.critic_local=", self.critic_local)
