[image2]: /Users/parksoy/Desktop/deep-reinforcement-learning/p1_navigation/banana_POR_episode_534_withtitle.jpg "Plot_score"

# Report
##### A description of the implementation  

This project scopes to learn how to train an agent to navigate and collect bananas in a large, square world.  A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. The state space has 37 dimensions including the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.


# Methodology

The same version of Deep Q Learning (DQN) example in this nanodegree was used as published in

```
https://github.com/parksoy/Soyoung_Udacity_ND_DeepReinforcementLearning/blob/master/dqn/exercise/model.py
https://github.com/parksoy/Soyoung_Udacity_ND_DeepReinforcementLearning/blob/master/dqn/exercise/dqn_agent.py
```
and was integrated into Unity ML-Agents environment. The agent's actions are controlled by DQN learning process and reward is obtained from Unity environment.

(1) Learning algorithm  

DQN network is used for each qnetwork_local and qnetwork_target.

## TO DO:
*DQN's key features - such as the use of experience replay and soft updates at a minimum. Since you also made use of the epsilon-greedy action selection mechanism, you should also briefly describe that in your report, along with any other unique features of your implementation (such as the UPDATE_EVERY section of your agent's step() method). Although you described the size of your NN hidden layers, your hyperparameter description should include other hyperparameters you used (such as TAU and Epsilon at a minimum), including a brief description of the purpose of each hyper parameter and the values you chose in your implementation.

*You have correctly decoupled parameters being updated for learning from the ones being used to produce target values. Implementing the soft update process helps to smooth out individual learning batches by preventing large fluctuations in actions from being generated during learning. You have correctly applied the Tau hyper parameter to control your soft-update process in your agent's Learn() method.
*You are correctly using an experience replay buffer, storing (and later, sampling) experience tuples consisting of (state, action, reward, next_state, done) fields. Your agent's step() function is correctly saving new experience tuples as they are encountered and then randomly selecting experience tuples to learn from.
*Your agent's act() function includes an epsilon-greedy action selection mechanism to encourage exploratory behavior in the agent, particularly during the early episodes. You are correctly using the Epsilon hyper parameter to control this process in your agent's act() method.



(3) The model architectures for neural networks

The given neural network for DQN network was used as is, which consists of three hidden linear neural network layers:

```
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```

Each layer is forwarded and mapped through relu.


(2) The chosen hyperparameters

The same given hyperparameters are used as the following:
```
fc1_units=64,
fc2_units=64
```


# A plot of rewards

![Plot_score][image2]

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. In order for the environment to achieve at least score of 13, 434 episodes was needed to solve the environment.


# Ideas for Future Work

In order to improve the agent's performance (less number of episodes to be explored to achieve the same score of 13 or more), the following ideas are considered:

(1) The more complex network other than three linear neural networks can be considered.  
##TO DO:
For example,

(2) The hyperparameters other than number of units in each layer can be varied.
For example,  
