from collections import deque
import random
import torch
import numpy as np
import copy
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state): #, done
        """Add a new experience to memory."""
        e = (state, action, reward, next_state)
        self.memory.append(e)#return

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.memory, k=self.batch_size)
        print("NOW # of SAMPLED EXPERIENCE from total \
        memory %d : %d" %(len(self.memory), len(batch)))
        print("type(batch)", type(batch), len(batch), type(zip(*batch)))
        states, actions, rewards, next_states = zip(*batch)
        print("states=", type(states), len(states), states)

        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).float().to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)

        '''
        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        #dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        '''
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
