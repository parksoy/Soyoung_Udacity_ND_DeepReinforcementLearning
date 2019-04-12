%reset

#cwd is where runner_reacher.py is located
import os
cwd='/Users/parksoy/Desktop/deep-reinforcement-learning/p2_continuous-control/'
if not os.getcwd()==cwd:
    os.chdir('/Users/parksoy/Desktop/deep-reinforcement-learning/p2_continuous-control/')
print(os.getcwd())

from unityagents import UnityEnvironment
import numpy as np
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent_reacher import Agent
from nnmodels_reacher import Actor, Critic
from collections import deque
from config_settings import Args
from replay_buffer import ReplayBuffer
from plotter_reacher import plot_scoreOverEpisodes
%matplotlib inline

%load_ext autoreload
%autoreload 2

####################################
#1.Initiate UnityEnvironmemt, call in all settings
####################################
'''
UnityTimeOutException: The Unity environment took too long to respond. Make sure that :
	 The environment does not need user interaction to launch
	 The Academy and the External Brain(s) are attached to objects in the Scene
	 The environment and the Python interface have compatible versions.
'''
env = UnityEnvironment(file_name='Reacher_multi.app', no_graphics=True)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

args=Args()

####################################
#2. EDA:the State and Action Spaces
####################################
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

####################################
#3.Take Random action
####################################
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: %.2f' %(np.mean(scores)))

####################################
#4. Train with DDPG
####################################
#nd array states.shape-(20, 33) len(rewards)=20. rewards is list, action_size=4
env_info = env.reset(train_mode=True)[brain_name]
agent = Agent(state_size=state_size, action_size=action_size, random_seed=2) #state_size 33, action_size 4
states = env_info.vector_observations


def train_storeExperience_getScore(n_episodes=args.num_episodes,
                                    max_t=args.max_steps,
                                    print_every=args.print_every):
    scores_deque = deque(maxlen=print_every)
    scores = []
    #agent.t_step=0
    print("Will start iteration of %d..." %args.num_episodes)

    for i_episode in range(1, n_episodes+1):
        print("\n####################")
        print("episode #=", i_episode)
        print("####################")
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = 0
        for t in range(max_t):
            print("  step#=",t)
            action = agent.act(states) #(20,33)
            action = np.clip(action, -1, 1)
            env_info = env.step([action])[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, action, rewards, next_states, dones)
            states = next_states
            score = score + np.average(rewards)
            if dones[0]:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'p2_checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'p2_checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    plot_scoreOverEpisodes(scores)
    print("Final scores=",scores)
    print("Done!")
    return scores

scores = train_storeExperience_getScore()


#############################################
#5. Evaluate a Trained Smart Agent
#############################################
agent.actor_local.load_state_dict(torch.load('p2_checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('p2_checkpoint_critic.pth'))

env_info = env.reset(train_mode=False)[brain_name]
states = env_info.vector_observations
for t in range(10): #range(200)
    action = agent.act(states, add_noise=False) #expected np.ndarray (got dict)
    action = np.clip(action, -1, 1)
    env_info = env.step([action])[brain_name]
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    if dones:
        print("action=\n",action,"\nrewards=\n",rewards,"\ndones=\n",dones)
        break

#env.close()
