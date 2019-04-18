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

    def add(self, states, actions, rewards, next_states): #, done
        """Add a new experience to memory."""
        experience = (states, actions, rewards, next_states)
        namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states"])

        self.memory.append(experience)#return

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.memory, k=self.batch_size)
        print("batch=",type(batch),len(batch)) #batch= <class 'list'> 8

        states, actions, rewards, next_states = zip(*batch)
        print("after unzip,states=" ,type(states), len(states)) #<class 'tuple'> 8
        print("states=", type(states), len(states)) #<class 'tuple'> 8
        print("actions=",type(actions))  # <class 'tuple'> 8 actions= (array([[0.07954566, 0.20505555, 0.145873  , 0.07304925],   [0.29717416, 0.31180978, 0.15491767, 0.11667927]], dtype=float32))
        print("actions=",actions) #(array([[0.160445


        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).float().to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)

        print("After unzip batch and torch cat:", type(states), len(states))

        '''
        print("[e.actions for e in batch]=", [e for e in batch])

        tmp_actions=torch.from_numpy(np.vstack([e.actions for e in batch if e is not None]))
        print("tmp_actions=",tmp_actions)

        states = torch.from_numpy(np.vstack([e.states for e in batch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in batch if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in batch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in batch if e is not None])).float().to(device)
        #dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(device)
        '''
        experiences=(states, actions, rewards, next_states)
        return  experiences #, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
