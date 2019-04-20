from collections import deque
import random
import torch
import numpy as np
import copy
from collections import namedtuple, deque
from config_settings import Args

args=Args()
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
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "done"]) #


    def add(self, states, actions, rewards, next_states, done): #
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, done) #
        self.memory.append(e)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.memory, k=self.batch_size)
        print("batch=",type(batch),len(batch)) #batch= <class 'list'> 8


        print("[e.actions for e in batch]=", [e._fields for e in batch])
        '''
        type(e)=
        [<class 'replay_buffer.Experience'>,
        <class 'replay_buffer.Experience'>,...

        [Experience(state=array([[ 0.00000000e+00,
        -4.37113883e-08,  0
        '''

        '''
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).float().to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)


        '''


        states = torch.from_numpy(np.vstack([e.states for e in batch if e is not None])).float().to(args.device)
        actions = torch.from_numpy(np.vstack([e.actions for e in batch if e is not None])).float().to(args.device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in batch if e is not None])).float().to(args.device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in batch if e is not None])).float().to(args.device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(args.device)

        experiences=(states, actions, rewards, next_states, dones)
        return  experiences #

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
